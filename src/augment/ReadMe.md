# Augment

针对透视增强数据。

> 标签必须是 PascalVOC。

## 使用方法

先安装依赖。

```shell
pip install -r package-list.txt
```

像下面这样从命令行传入参数，或使用`augment.py`末尾`if`中的`args = parser.parse_args([…])`。
```shell
> python augment.py Annotations/ Images/ Annotations-out/ Images-out/
```

这四个路径的更多解释请见`--help`。

```shell
> python augment.py --help
usage: augment.py [-h] ……

针对透视增强数据

positional arguments:
  input_annotations   原数据标签所在目录
……
……
```

## 具体干了什么？

对每张照片应用仿射变换，并一同修改标签中的框。

- 仿射变换

  随机移动照片四角，范围 -5% ~ +5%（相对照片长宽而言）。

- 标签中的框
  1. 先正常变换，把原来的框变为不规则四边形。
  2. 把各边中点形成的矩形作为新的框。
