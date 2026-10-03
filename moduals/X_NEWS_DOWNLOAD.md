# X AI 新闻数据下载

下载最近 720 小时的 X AI 新闻事件、关联证据帖子、作者资料和外部链接 metadata。

## 凭据

在项目根目录创建 `.env`：

```text
X_BEARER_TOKEN=你的_X_API_Bearer_Token
```

下载器不会把密钥写入输出文件或请求 URL。

## 运行

```powershell
python .\download_x_ai_news.py
```

请使用项目根目录入口。直接运行 `moduals` 内的文件会让该目录中的
`types.py` 与 Python 标准库同名模块发生冲突。

默认输出到：

```text
data/x_ai_news_30d/
```

脚本首先保存原始 API JSON，然后生成：

```text
normalized/news_stories.csv
normalized/evidence_posts.csv
normalized/authors.csv
normalized/story_post_map.csv
normalized/linked_articles.csv
normalized/reaction_posts.csv
collection_manifest.json
```

如果下载中断，重新运行会复用已经保存的原始响应。使用 `--refresh` 可以强制重新下载。
使用 `--offline` 可以在不发出新 API 请求的情况下，把已下载的原始响应整理成 CSV。

`reaction_posts.csv` 第一轮只创建表头。完整 30 天的回复与引用帖子需要在确认 X Full-Archive Search 权限后单独下载；覆盖状态记录在 manifest 中。
