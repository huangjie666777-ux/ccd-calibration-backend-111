# CCD 标定纯后端

使用 FastAPI 接收一批二维单色 FITS 曝光，构建主偏置、每秒暗电流和主平场，并输出可供测光使用的浮点科学帧。原始上传文件只在内存中读取，不会被修改。

## 接口

启动：

```bash
.venv/bin/uvicorn ccd_calibration.api:app --host 127.0.0.1 --port 8000
```

提交批次：`POST /calibrate/batch`，`multipart/form-data`：

- `files`：可重复字段，所有原始 FITS，主 HDU 必须是二维数值图像。
- `roles`：可重复字段，与文件顺序一一对应，取值为 `bias`、`dark`、`flat`、`science`。
- `parameters`：可选 JSON，例如 `{"sigma":3.0,"maxiters":5}`。

成功返回 `application/zip`。任何文件级或批级错误都返回 `422`，响应中的 `detail.file_errors` 按文件名说明原因，并且不交付部分结果。

## FITS 头约定

所有帧都必须包含：

- `INSTRUME`：仪器名，非空字符串。
- `CCDTEMP`：探测器温度，数值且有限。
- `GAIN`：增益，正数值。
- `XBINNING`、`YBINNING`：正整数分箱。
- `EXPTIME`：秒；bias 允许零和非负值，dark、flat、science 必须为正。
- `SATURATE`：饱和上限；原始像素超过该值标记为坏像素。

`flat` 与 `science` 还必须包含相同的非空 `FILTER`。整批还必须具有相同仪器、温度、增益、分箱和二维尺寸。输入主 HDU 的物理像素值以 `float64` 读取；非有限输入值会被排除并在科学帧掩膜中标记。

## 计算方法

1. 主偏置：跨 bias 逐像素排除坏输入和非有限值，执行 sigma 裁剪均值。
2. 每秒暗电流：每张 dark 先减去主偏置，再除以该帧自己的 `EXPTIME`，然后逐像素 sigma 裁剪均值。偏置本身不随曝光时间缩放。
3. 主平场：每张 flat 减去主偏置和 `dark_rate * EXPTIME`；用有限且为正的有效像素中位数归一化，再逐像素 sigma 裁剪均值；组合帧再次按有效像素中位数归一化。
4. 科学帧：`(science - master_bias - master_dark_rate * EXPTIME) / master_flat`。平场零、负或非有限的位置输出 `NaN`；科学帧中的负校正值保留。

暗电流模型假设在所选曝光时间和工作温度范围内线性、稳定，且 dark 与 science/flat 的探测器状态一致。非线性、余辉、随时间漂移或温度变化明显的数据不适合仅用该线性模型校正。

## 掩膜

每个科学帧是一个浮点 FITS，包含：

- 主 HDU：校准结果，坏像素为 `NaN`，保留有效负值。
- `BADMASK` 扩展：位掩膜，`1` 为非有限输入，`2` 为超过 `SATURATE`，`4` 为平场非正或无效，`8` 为偏置/暗电流位置无有效样本。
- `MASK_REASONS` 表：掩膜位说明。

ZIP 还包含 `master_bias.fits`、`master_dark_current_per_second.fits`、`master_flat.fits`、`source_manifest.csv` 和 `processing_metadata.json`。输出头保留原始观测头，并通过 `CAL*` 关键字与 `HISTORY` 记录步骤和参数。

## 示例与测试

生成小型合成批次：

```bash
.venv/bin/python examples/generate_example_batch.py
```

运行测试：

```bash
.venv/bin/python -m compileall -q ccd_calibration examples tests
.venv/bin/pytest -q
```

服务启动后可使用：

```bash
curl -f -X POST http://127.0.0.1:8000/calibrate/batch \
  -F 'files=@examples/batch/bias_01.fits;type=application/fits' \
  -F 'files=@examples/batch/bias_02.fits;type=application/fits' \
  -F 'files=@examples/batch/bias_03.fits;type=application/fits' \
  -F 'files=@examples/batch/dark_01.fits;type=application/fits' \
  -F 'files=@examples/batch/dark_02.fits;type=application/fits' \
  -F 'files=@examples/batch/dark_03.fits;type=application/fits' \
  -F 'files=@examples/batch/flat_01.fits;type=application/fits' \
  -F 'files=@examples/batch/flat_02.fits;type=application/fits' \
  -F 'files=@examples/batch/flat_03.fits;type=application/fits' \
  -F 'files=@examples/batch/science_01.fits;type=application/fits' \
  -F 'files=@examples/batch/science_02.fits;type=application/fits' \
  -F 'roles=bias' -F 'roles=bias' -F 'roles=bias' \
  -F 'roles=dark' -F 'roles=dark' -F 'roles=dark' \
  -F 'roles=flat' -F 'roles=flat' -F 'roles=flat' \
  -F 'roles=science' -F 'roles=science' \
  -F 'parameters={"sigma":3.0,"maxiters":5};type=application/json' \
  -o ccd_calibration_package.zip
```
