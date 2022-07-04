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

> 输入输出目录可以相同，但那样不容易撤销更改，建议试试再这么做。

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

- 重命名

  照片的文件名会被加上后缀，例如`xyc000.jpg` → `xyc000-augment-2022-07-04-23_13_59.jpg`。标签自身的文件名以及其内容中的文件名也会如此重命名。

## 潜在的问题

- 需要联系**原数据**的照片和标签，我并未读取标签内容，只是按文件名对应。

  如果照片和标签的文件名不一样，`augment.py`会炸。

- 原数据的标签中的`<path>`可能错误，而制造出的`<path>`总是指向实际制造出的照片。

  > 原数据的`<path>`可能错误，是因为打标签的电脑可能是另外的电脑。

  如果后续不关心`<path>`，这不会引发任何问题。

- 文件名只精确到秒。

  如果一秒内运行了多次`augment.py`，可能导致照片、标签不一致。
