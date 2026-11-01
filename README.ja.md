[English](README.md)

# sarashina-bot

メンションされると [Sarashina](https://api.sarashina.ai/v1) のチャット API で返事をする Discord bot です。

- モデルは [Exa](https://exa.ai) で web を検索できます（`web_search` ツール。全員に見せる）。
- `DISCORD_ALLOWED_USERS` に入っている人には `run_command` ツールも見せます。別の Pod のサンドボックスに HTTP でコマンドを送り、bash で実行します。
- 同じく `DISCORD_ALLOWED_USERS` の人は `/model [name]` で `SARASHINA_MODELS` のモデルに切り替えられます（返事は本人にだけ見え、切り替えはプロセスが再起動するまで続きます）。
- `/effort level` では `reasoning_effort`（`low` / `medium` / `high`）を指定できます。受け付けるのは `sarashina3.2-mini-thinking` だけなので、ほかのモデルのときは断り、送るのもこのモデルを選んでいるあいだだけです。指定は `/model` で切り替えても残ります。考えているうちに出力の上限を使い切って回答が空になったときは、「回答が出せませんでした。effort を下げてください。」と返します。
- 同じ人は `/translate source text` で、Sarashina の翻訳 API を使って日本語と英語を翻訳できます。`source` に `ja` か `en` を選ぶと、もう一方の言語に訳します。`text` は 5,000 文字までです。返事は全員に見え、中身は訳文だけです（2,000 文字を超えたら分けて送ります）。失敗したときは「翻訳に失敗しました。」と返します。翻訳は会話の文脈には入りません。

- テキストチャンネルに arXiv URL を貼ると、メンションなしで `arXiv:{id}` スレッドを作ります。
  索引ができると 1 段落ずつ投稿し、その途中でもスレッドに質問すれば論文の抜粋をもとに答えます。
  空でないメッセージは返信も含めてすべて質問として扱い、誰でも送れます。ターミナルのツールは使いません。
  `DISCORD_ALLOWED_USERS` の人は bot の段落を右クリック（モバイルでは長押し）→ アプリ → 日本語に翻訳で選ぶと、全員に見える日本語訳を受け取れます。
  スラッシュコマンドは返信先のメッセージを受け取れず、返信中に `/translate` を入力しても登録済みコマンドが開くため、段落翻訳には専用のメッセージコンテキストメニューを使います。
  段落翻訳は再起動後も使えます。質問に使う索引はメモリ上だけにあるので、再起動後に質問を再開するには URL をもう一度貼ってください。

設計の詳細や Sarashina API のクセは [SPEC.md](SPEC.md) にまとめてあります。

## 構成

イメージは 1 つで、プロセスが 2 つあります。

| プロセス | コマンド | 役割 |
| --- | --- | --- |
| bot | `python -m sarashina_bot`（イメージの既定） | Discord につなぎ、Sarashina と Exa を呼び、`run_command` をサンドボックスに回す |
| サンドボックス | `python -m sarashina_bot.sandbox` | 8080 番で待つ HTTP サーバー。`POST /run` に `{"command": str}` を送ると `{"exit_code", "timed_out", "output"}` が返る |

サンドボックスの制約:

- UID 10001 で動き、作業ディレクトリと `HOME` は `/work`。書き込めるのは `/work` と `/tmp`
- コマンドは新しいセッションで `bash -c` に渡し、30 秒たったらプロセスグループごと SIGKILL する
- stdout と stderr をまとめた出力の先頭 4,000 バイトを返す
- コマンドに渡す環境変数は `PATH`・`HOME`・`LANG`・`TMPDIR` だけで、シークレットは持たない
- 認証はない。デプロイ先では NetworkPolicy で bot の Pod からしか届かないようにしている

1 回のメンションでモデルを呼ぶのは多くて `MAX_TOOL_ROUNDS + 1` 回です。ツール付きで最大 `MAX_TOOL_ROUNDS` 回回したあと、最後はツールを外して文章で答えさせます。

### 会話の文脈

メンションされると、チャンネルの直近 `CONTEXT_MESSAGE_COUNT` 件を取ってきて、メンションした人の発言（`user`）と、その人への bot の返事（長文を分けて送った続きも含む。`assistant`）だけを残し、ほかの人の発言は捨てます。先頭が `assistant` になったら、それは捨てて `user` から始めます。システムプロンプトの末尾には「話しかけているのは〇〇さんです。」を足します。履歴は保存せず、毎回 Discord から読みます。

## 設定

設定はすべて環境変数で渡します（`src/sarashina_bot/config.py`）。

| 変数 | 必須 | 既定値 | 意味 |
| --- | --- | --- | --- |
| `DISCORD_TOKEN` | ○ | | Discord bot のトークン |
| `SARASHINA_API_KEY` | ○ | | Sarashina の API キー |
| `EXA_API_KEY` | ○ | | `web_search` に使う Exa の API キー |
| `DISCORD_ALLOWED_USERS` | | 空 | `run_command`・`/model`・`/effort`・`/translate` とメッセージコンテキストメニュー「日本語に翻訳」を使える Discord ユーザー ID（カンマ区切り） |
| `SARASHINA_MODEL` | | `sarashina3-mini` | 起動時のモデル。`SARASHINA_MODELS` に入っていないと起動しない |
| `SARASHINA_MODELS` | | `SARASHINA_MODEL` の値 | `/model` の選択肢（カンマ区切り） |
| `SANDBOX_URL` | | `http://sandbox:8080` | サンドボックスのベース URL |
| `MAX_TOOL_ROUNDS` | | `2` | 1 回のメンションでツールを回す上限。`MAX_TOOL_ROUNDS + 1` は `RATE_LIMIT_RPM` 以下 |
| `RATE_LIMIT_RPM` | | `30` | Sarashina へのリクエストを 1 分あたり何回までにするか（クライアント側の制限）。超えたときや Sarashina から 429 が返ったときは、「{x}秒後にもう一度送信してください。」と返信する。Sarashina は翻訳をチャットとは別に数えるので、`/translate` には同じ値で別のリミッタを置いている |
| `CONTEXT_MESSAGE_COUNT` | | `10` | 絞り込む前に読むチャンネルの直近メッセージ数 |
| `SYSTEM_PROMPT` | | 組み込みの短い日本語のプロンプト | システムプロンプト |
| `LOG_LEVEL` | | `INFO` | Python のログレベル（サンドボックスもこれを読む） |

必須の変数がないとき、`SARASHINA_MODEL` が `SARASHINA_MODELS` にないとき、`MAX_TOOL_ROUNDS + 1 > RATE_LIMIT_RPM` のときは bot が起動を止めます。サンドボックスが読むのは `LOG_LEVEL` だけです。

## 開発

[uv](https://docs.astral.sh/uv/) を使います。Python のバージョンは `.python-version` で固定しています。

```sh
uv sync
uv run pytest
```

手元で bot を動かすとき:

```sh
export DISCORD_TOKEN=... SARASHINA_API_KEY=... EXA_API_KEY=...
export DISCORD_ALLOWED_USERS=123456789012345678   # なくてもいい
export SANDBOX_URL=http://localhost:8080          # run_command を試すときだけ
uv run python -m sarashina_bot
```

サンドボックスの作業ディレクトリ `/work` はコードに直書きなので、あらかじめ存在している必要があります（Kubernetes では emptyDir をマウントしている）。ないとどのリクエストも 500 になります。手元でそのまま動かすとコマンドが自分のユーザーで走るので、コンテナの中で動かすほうが安全です。

```sh
podman run --rm -p 8080:8080 --tmpfs /work:mode=1777 \
  localhost/sarashina_bot:dev python -m sarashina_bot.sandbox
curl -s localhost:8080/run -d '{"command": "id; pwd"}'
```

## ビルド

```sh
podman build -t localhost/sarashina_bot:<tag> .
```

`docker build` でもビルドできます。Dockerfile が `RUN --mount=type=cache` を使うので、Docker では BuildKit が要ります（今のバージョンなら既定で有効）。

CI（`.github/workflows/ci.yml`）は pull request と `main` への push でテストと `docker build .` を回します。イメージの push はしません。

## デプロイ

Kubernetes のマニフェストとデプロイスクリプトは private の homelab リポジトリの [`sarashina-bot/`](https://github.com/koh11235813/homelab/tree/main/sarashina-bot) に置いています。`sarashina-bot/deploy.sh <この clone のパス>` でイメージをビルドしてロールアウトします。手順はそちらを見てください。

## ライセンス

PyMuPDF に依存しているため、AGPL-3.0-only を採用しています。全文は [LICENSE](LICENSE) を参照してください。

Copyright (C) 2026 koh11235813
