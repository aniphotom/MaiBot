#!/bin/bash
# 在部署机上切换 core。先拉取并固定 digest，失败时恢复原来的镜像和 compose/.env。
# 不执行 down、不删卷、不 prune。只重建 core，不动其他服务。
set -euo pipefail

: "${DEPLOY_DIR:?}"
: "${WANT_IMAGE:?}"

cd "$DEPLOY_DIR"
mkdir -p .deploy-rollback
exec 9>.deploy-rollback/lock
if ! flock -n 9; then
  echo "已有部署在进行，本次不改动运行中的服务" >&2
  exit 1
fi

if [[ ! "$WANT_IMAGE" =~ ^ghcr\.io/aniphotom/maibot:(latest|sha-[0-9a-f]{40})$ ]] \
  && [[ ! "$WANT_IMAGE" =~ ^ghcr\.io/aniphotom/maibot@sha256:[0-9a-f]{64}$ ]]; then
  echo "拒绝的镜像引用：${WANT_IMAGE}" >&2
  exit 1
fi

if [ ! -f docker-compose.yml ]; then
  echo "服务器上没有 docker-compose.yml。不会上传本仓库的编排。" >&2
  exit 1
fi

changed=0
script_ok=0

was_target_tag=0
if docker inspect maim-bot-core >/dev/null 2>&1; then
  old_id="$(docker inspect --format '{{.Image}}' maim-bot-core)"
  old_ref="$(docker image inspect --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' "$old_id")"
  printf '%s\n' "$old_id" > .deploy-rollback/image-id
  printf '%s\n' "$old_ref" > .deploy-rollback/image-ref
  if [[ "$WANT_IMAGE" != *@* ]] \
    && docker image inspect --format '{{range .RepoTags}}{{println .}}{{end}}' "$old_id" | grep -Fx "$WANT_IMAGE" >/dev/null; then
    was_target_tag=1
  fi
  echo "当前 core ${old_ref}（${old_id}）"
else
  : > .deploy-rollback/image-id
  : > .deploy-rollback/image-ref
  old_id=""
  old_ref=""
  echo "当前没有运行中的 core"
fi
if [ -f docker-compose.yml ]; then
  cp -a docker-compose.yml .deploy-rollback/docker-compose.yml
else
  rm -f .deploy-rollback/docker-compose.yml
fi
rm -f .deploy-rollback/env .deploy-rollback/env.present .deploy-rollback/env.absent
if [ -f .env ]; then
  cp -a .env .deploy-rollback/env
  : > .deploy-rollback/env.present
else
  : > .deploy-rollback/env.absent
fi

core_image() {
  awk '
    $0 ~ /^  core:/ { in_core=1; next }
    in_core && $0 ~ /^  [^[:space:]]/ { in_core=0 }
    in_core && $0 ~ /^[[:space:]]*image:/ {
      gsub(/["'\'']/, "", $2)
      print $2
      exit
    }
  '
}

restore_files() {
  if [ -f .deploy-rollback/docker-compose.yml ]; then
    cp -a .deploy-rollback/docker-compose.yml docker-compose.yml
  else
    rm -f docker-compose.yml
  fi
  if [ -f .deploy-rollback/env.present ]; then
    cp -a .deploy-rollback/env .env
  elif [ -f .deploy-rollback/env.absent ]; then
    rm -f .env
  fi
}

rollback() {
  echo "部署失败，正在恢复原来的 core" >&2
  restore_files
  if [ ! -s .deploy-rollback/image-id ]; then
    docker compose stop core || true
    docker compose rm -f core || true
    if docker inspect maim-bot-core >/dev/null 2>&1; then
      echo "回滚失败：原来没有 core，但本次容器还在" >&2
      return 1
    fi
    echo "原来没有运行中的 core。已移除本次 core 容器，没有删除数据卷。" >&2
    return 0
  fi
  old_id="$(cat .deploy-rollback/image-id)"
  old_ref="$(cat .deploy-rollback/image-ref)"
  if ! docker image inspect "$old_id" >/dev/null 2>&1; then
    case "$old_ref" in
      *@sha256:*)
        docker pull "$old_ref" || {
          echo "回滚失败：拉取原镜像 ${old_ref} 失败" >&2
          return 1
        }
        ;;
      *)
        echo "回滚失败：本机没有原镜像 ${old_id}，记录 ${old_ref} 也不是可拉取的 digest" >&2
        return 1
        ;;
    esac
  fi
  config_text="$(env -u MAIBOT_IMAGE docker compose config)"
  resolved="$(core_image <<< "$config_text")"
  resolved="${resolved%\"}"
  resolved="${resolved#\"}"
  case "$resolved" in
    "" )
      echo "回滚失败：恢复后的 compose 没有解析出 image" >&2
      return 1
      ;;
    *@*) ;;
    *) docker tag "$old_id" "$resolved" ;;
  esac
  if ! env -u MAIBOT_IMAGE docker compose up -d --no-deps --force-recreate core; then
    echo "回滚失败：无法按原来的 compose/.env 重新启动 core" >&2
    return 1
  fi
  now="$(docker inspect --format '{{.Image}}' maim-bot-core)"
  if [ "$now" != "$old_id" ]; then
    echo "回滚失败：core 现在是 ${now}，不是原来的 ${old_id}（${old_ref}）" >&2
    return 1
  fi
  echo "已恢复 core 到 ${old_ref}（${old_id}）" >&2
  return 0
}

finish() {
  status=$?
  trap - EXIT
  if [ "$script_ok" = 1 ]; then
    exit 0
  fi
  if [ "$changed" = 1 ]; then
    rollback || echo "回滚未完成，部署保持失败" >&2
  fi
  if [ "$status" -eq 0 ]; then
    status=1
  fi
  exit "$status"
}
trap finish EXIT

echo "拉取 ${WANT_IMAGE}"
docker pull "$WANT_IMAGE"
digests="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$WANT_IMAGE")"
pinned="$(awk '/^ghcr\.io\/aniphotom\/maibot@sha256:[0-9a-f]{64}$/ {print; exit}' <<< "$digests")"
if [ -z "$pinned" ]; then
  if [ "$was_target_tag" = 1 ]; then
    docker tag "$old_id" "$WANT_IMAGE" || true
  fi
  echo "没有得到 ghcr.io/aniphotom/maibot 的 digest，停止。运行中的服务未切换。" >&2
  exit 1
fi
echo "固定 ${pinned}"
target_id="$(docker image inspect --format '{{.Id}}' "$pinned")"

changed=1
touch .env
{ grep -v -e '^MAIBOT_IMAGE=' -e '^MAIBOT_IMAGE_TAG=' .env || true; } > .env.tmp
printf 'MAIBOT_IMAGE=%s\n' "$pinned" >> .env.tmp
mv .env.tmp .env
current="$(core_image <<< "$(MAIBOT_IMAGE="$pinned" docker compose config)")"
if [ "$current" != "$pinned" ]; then
  tmp="$(mktemp)"
  if ! awk -v img="$pinned" '
    $0 ~ /^  core:/ { in_core=1 }
    in_core && $0 ~ /^  [^[:space:]]/ && $0 !~ /^  core:/ { in_core=0 }
    in_core && !done && $0 ~ /^[[:space:]]*image:/ {
      print "    image: " img
      done=1
      next
    }
    { print }
    END { if (!done) exit 2 }
  ' docker-compose.yml > "$tmp"; then
    rm -f "$tmp"
    echo "现有 compose 的 core 没有可替换的 image 行。未上传仓库编排。" >&2
    exit 1
  fi
  mv "$tmp" docker-compose.yml
fi
MAIBOT_IMAGE="$pinned" docker compose up -d --no-deps --force-recreate core

running_id="$(docker inspect --format '{{.Image}}' maim-bot-core)"
if [ "$running_id" != "$target_id" ]; then
  echo "切换后 core 仍是 ${running_id}，目标是 ${target_id}" >&2
  exit 1
fi

ready=0
i=1
while [ "$i" -le 90 ]; do
  running_id="$(docker inspect --format '{{.Image}}' maim-bot-core)"
  if [ "$running_id" != "$target_id" ]; then
    echo "健康检查期间 core 镜像变为 ${running_id}" >&2
    exit 1
  fi
  if curl -fsS -o /dev/null http://127.0.0.1:18001; then
    echo "core 镜像已匹配目标，本机 18001 在 ${i}s 内响应"
    ready=1
    break
  fi
  i=$((i + 1))
  sleep 1
done
if [ "$ready" != 1 ]; then
  echo "90 秒内本机 18001 未响应。最近日志：" >&2
  docker compose logs --tail 50 core >&2 || true
  exit 1
fi
script_ok=1
