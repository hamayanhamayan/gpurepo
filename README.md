# gpurepo

## プロファイル方法

```
$ /usr/local/cuda/bin/nvprof --events shared_load_replay,shared_store_replay ./～.out
```

でバンクコンフリクトがプロファイリングできる