# 自有镜像：手动构建，再手动部署

镜像名只有一个：`ghcr.io/aniphotom/maibot`。构建和部署都用它。

当前 `main` 的应用源码与现网官方 1.3.9（`ac6ed61`）相同。自有镜像重新构建后，digest 不会等于官方 `sengokucola/maibot@sha256:90760b3060af26549f7cd075521522d3f26f7504d444a6e53629ef8341977d05`。

## 构建

Actions → **Build and Push (Self)** → Run workflow。只在手动触发时构建 `linux/amd64` 并推送，不会部署。

- 日志 `tag=` 是交给推送步骤的 `ghcr.io/aniphotom/maibot:sha-<本次 github.sha>`
- 日志 `by_digest=` 是同一镜像的 `ghcr.io/aniphotom/maibot@sha256:...`
- `github.sha` 是本次手动运行所选自的提交，不是现网 `ac6ed61`，除非当时选自的就是它
- 登录用 `GITHUB_TOKEN`，不新增 PAT。仓库 Actions 若把令牌限制为只读，这里的 `packages: write` 不会生效

## 部署

构建完成并确认日志后，Actions → **Manual Deploy**。`tag` 填下面之一：

- `sha-<40位提交>`
- `sha256:<64位>`
- 或日志中的完整 `ghcr.io/aniphotom/maibot:sha-...` / `ghcr.io/aniphotom/maibot@sha256:...`

非法值、`latest`、官方仓库名在 SSH 之前失败。`pull` 和 `up` 使用本次校验过的 `MAIBOT_IMAGE`，空值会让 compose 停住，core 没有隐式 `latest`。流程仍用原有 SSH、目录 `project/maibot`、同步 `docker-compose.yml` 和健康检查。健康检查只看 `18001` 是否响应，不核对镜像 digest。`.env` 只改 `MAIBOT_IMAGE`。卷挂载不改：`docker-config/mmc`、`data/MaiMBot`、插件数据、表情、`plugins`、日志、`depends-data`。

宿主机 `./data/MaiMBot/plugins` 独立挂载。构建核心不会更新插件，身份和群 @ 的既有补丁留在该目录。

此部署流程没有备份步骤。回滚：用上一个 `sha-` tag 或 `sha256:` digest 再跑一次 Manual Deploy。回到现网官方镜像时，把 compose 的 `image` 改回上面的官方 digest，不要用本流程的 `tag` 输入去指官方仓库。

推送本分支或打开 PR 不会构建、不会部署。对这两个工作流点 Run 才会连 GHCR 或生产。镜像在有人手动运行 Build and Push (Self) 且成功之前不能拉取。合并到 `main` 仍会触发原有 `docker-image-main.yml`（Docker Hub，含 arm64）；本说明不改那个文件。
