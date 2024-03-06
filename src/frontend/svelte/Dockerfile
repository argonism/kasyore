# ビルド用
FROM node:18.12.1-slim as builder

WORKDIR /app

## パッケージをインストール
COPY package.json ./
COPY package-lock.json ./
COPY tsconfig.json ./
RUN npm ci

COPY . .

RUN npm run build

# 実行用
FROM node:18.12.1-slim

WORKDIR /app

## ビルド用のレイヤからコピーする
COPY --from=builder /app/build ./build
COPY --from=builder /app/package.json .
COPY --from=builder /app/node_modules ./node_modules

## Svelteが動く5173ポートを開けておく
EXPOSE 5173

CMD ["node", "./build"]
