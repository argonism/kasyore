-- DB作成
CREATE DATABASE kasyore; 

-- 作成したDBへ切り替え
\c kasyore

-- テーブル作成
CREATE TABLE  esa_docs (
  number VARCHAR(10),
  full_name TEXT,
  wip BOOLEAN,
  body_md TEXT,
  body_html TEXT,
  created_at timestamp with time zone,
  updated_at timestamp with time zone,
  url TEXT,
  created_by VARCHAR(20),
  PRIMARY KEY (number)
);
