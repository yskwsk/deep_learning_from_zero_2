# ゼロから作る Deep Learing - 自然言語処理編

- 書籍[『ゼロから作る Deep Learning - 自然言語処理編』](https://www.oreilly.co.jp/books/9784873118369/)
- [書籍のサポートページ(Github)](https://github.com/oreilly-japan/deep-learning-from-scratch-2)

## 環境
- Python: 3.13.12

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

実行例
```bash
cd src/ch03

# mypyによる静的型チェック
./mypy_run mnist_show.py

# 実行
./python_run mnist_show.py
```