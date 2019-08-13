 # alcon_team3

## １．ファイル内容

### ディレクトリ

- srcs
  - split_img.py        画像を３等分する
  - train.py       いち文字ずつ画像を学習させる．   
  - utils.py     かなをunicodeに変換したりなどなどの関数が入っている．
- csv_files
  - annotataions.csv
  - label.csv
  - test_prediction.csv
- CNN_models 学習実行ファイル
  - a.h5
- main.py 実行ファイル.テスト対象の画像を3等分し，いち文字ずつの画像を予測する