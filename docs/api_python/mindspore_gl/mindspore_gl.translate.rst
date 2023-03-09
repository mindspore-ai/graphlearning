mindspore_gl.translate
======================

.. py:function:: mindspore_gl.translate(obj, method_name: str, translate_path: None or str = None)

    将顶点中心代码转换为MindSpore可理解代码。

    翻译后，将在`/.mindspore_gl`中生成一个新函数。原方法将被此函数替换。

    参数：
        - **obj** (Object) - 翻译对象。
        - **method_name** (int) - 要转换的方法的名称。
        - **translate_path** (int) - 构造文件的保存路径。
