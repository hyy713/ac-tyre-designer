# AC CSP Tyre Designer

一个面向 Assetto Corsa + CSP Extended Physics 的轮胎设计桌面工具。它允许定义紧凑型纯滑移 Magic Formula、绘制不同载荷下的曲线、反向拟合 CSP 参数，并导出一个可编辑的 AC 轮胎数据包。

## 功能

- 分别定义侧向和纵向 Magic Formula：`B/C/E/μ0/载荷指数/水平与垂直偏移`
- 所有数值框都带上下箭头，支持按住连续调整和鼠标滚轮，并实时更新曲线
- 绘制多载荷 `Fy-侧偏角`、`Fx-滑移率`、`μ-Fz` 和 `侧偏刚度-Fz`
- 可编辑并导出 `RELAXATION_LENGTH`、`FLEX`、`FLEX_GAIN` 和 `FRICTION_LIMIT_ANGLE`
- 轮胎惯量、阻尼和垂向刚度使用独立可编辑值，不再根据 `FZ0` 和半径自动推导
- 可直接导入本项目的 UniTire 实测 JSON，并转换为可编辑 Magic Formula/AC 参数
- 使用 AC 刷模型正向公式进行有界非线性最小二乘反向拟合
- 叠加显示目标 MF 与 CSP 代理曲线，并报告 R²/RMSE
- 保存/打开 JSON 设计
- 导出 `tyres.ini`、磨损 LUT、温度性能 LUT 和拟合报告

## 安装与启动（Windows）

最简单的方法是双击 `start.bat`。它会自动创建虚拟环境、安装缺失依赖并启动程序。

也可以在 PowerShell 中手动执行：

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

如果默认 PyPI 连接较慢，可以指定可用的镜像，例如：

```powershell
.\.venv\Scripts\python.exe -m pip install -i https://pypi.org/simple -r requirements.txt
```

## 使用

1. 输入轮胎尺寸、参考载荷、胎压，以及侧向/纵向 MF 参数。
2. 点击“更新曲线”检查目标响应。
3. 点击“拟合 CSP”，查看虚线和误差指标。
4. 点击“Export to AC”，选择 `tyres.ini` 的保存位置。程序会在同一目录生成关联 LUT 和设计 JSON。
   如果目标 `tyres.ini` 已存在，程序会先生成带时间戳的备份文件。
5. **备份目标车辆的整个 `data` 目录**，再把导出文件复制进去。
6. 使用 Content Manager/CSP 的轮胎调试工具验证不同载荷、胎温和胎压下的表现。

启动时默认填入的是随附实测模型 Hoosier 18.0 x 7.5 10 R25B (Item 43105) 的参数（来自 `tests/data/hoosier_43105_unitire_model.json`），打开后可直接点击“拟合 CSP”查看曲线效果。

要强制启用 CSP Extended Physics，目标车辆的 `car.ini` 应使用：

```ini
[HEADER]
VERSION=extended-2
```

原版 AC 的 `Tyre Tester` 不支持可靠显示 CSP 接管后的轮胎曲线。CSP 模式应使用 CSP Logger、原生 MoTeC 日志或 skidpad 实车遥测验证；程序自身的虚线用于检查导出参数的静态正向模型。

注意：AC 的磨损 LUT 输出使用百分数，新胎状态应接近 `100`，而不是归一化的 `1.0`。程序会在导出时检查这一点；错误的 `1.0` 会让 CSP 将新胎当作只剩 1% 状态，导致纵向和侧向力几乎为零。

前后轴当前使用同一套设计。需要不同前后胎时，可分别导出两次，然后把第二次导出的 `[FRONT_0]`/`[REAR_0]` 和对应热力学段合并到目标文件。

## 导出文件说明

导出会在所选目录生成 4 个文件。`tyres.ini` 中**每个参数都带中文注释**说明其作用：

| 文件 | 说明 |
| ---- | ---- |
| `tyres.ini` | 轮胎主文件（需 CSP Extended Physics），所有参数逐项注释 |
| `wear_curve.lut` | 磨损曲线：`x`=磨损量（0=全新），`y`=剩余性能百分比（新胎≈100） |
| `thermal_performance.lut` | 温度性能曲线：`x`=胎温(°C)，`y`=抓地力乘子（1.0=最佳工作温度） |
| `tyre_design.json` | 设计文件，可重新导入本工具编辑；顶层 `_notes` 键含全部字段的中文说明 |

`.lut` 是纯数据文件（AC/CSP 解析器不保证支持注释行），因此对它们的说明放在 `tyres.ini` 文件头的注释块中。

## 重要限制

CSP 轮胎求解器的完整实现并未公开。本工具不会声称能把标准 Pacejka 系数逐项“翻译”为 CSP 系数；它使用公开的 AC 刷模型重建公式进行反向拟合。生成结果仍须在 AC 中最终验证。热力、磨损、压力、外倾角、联合滑移和胎体结构的默认值不是由纯滑移 MF 唯一决定的，需要根据实测数据或目标车辆继续标定。

回正力矩 Mz 的拟合精度不足，且 AC/CSP 当前没有公开的 `tyres.ini` 字段可直接输入任意 Mz 曲线，因此本工具不再提供 Mz 拟合与预览。
