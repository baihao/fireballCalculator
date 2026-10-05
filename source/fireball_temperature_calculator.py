#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
火球温度计算器 — 时间与温度单位毫秒 / 开尔文

两条剖面可选（``profile``）：

1) ``reference_csv``（默认）：以内嵌参考温度曲线为基准（原 CSV 数据），
   使用单调保形分段三次插值（PCHIP）在全时间轴上贴合数据；可选用 ``reference_csv_path`` 覆盖为外部文件。

2) ``legacy``：原先「手工数字化 6 点 + 左侧三次多项式 + 右侧指数拖曳 + blend/c1」的解析近似。

工程仿真缩放（``temperature_baseline_scaled``）：

- **形状**：内嵌参考曲线（PCHIP，时间列仅作归一化参数
  ``u = t_ref / T_span``，0→1；**不**决定物理时长与峰值温度）。
- **物理时长**：``t_d``（ms）由 ``engineering_tab`` 火球持续时间公式（当量 W）。
- **峰值温度**：``1.3·T_eq``（``T_eq`` 由 ``engineering_tab`` 当量、含铝率、环境等计算）。
- 映射：``t_ref = (t / t_d) · T_csv,span``，再对 CSV 形状做幅值缩放至目标峰值。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

TEMP_OFFSET = 273.15

# 默认温度 CSV 标定当量（kg TNT）
REFERENCE_EQUIVALENT_KG = 100.0
# 温度曲线峰值相对 T_eq 的倍数
PEAK_TEMPERATURE_T_EQ_FACTOR = 1.3

# ---------------------------------------------------------------------------
# 内嵌参考温度曲线（原 fireball_temperature_reference_curve.csv，t=linspace(0,2000,1000)）
# 打包后无需依赖外部 CSV；若传入存在的 reference_csv_path 仍可覆盖。
# ---------------------------------------------------------------------------
_REFERENCE_CURVE_N = 1000
_REFERENCE_CURVE_T_SPAN_MS = 2000.0
_REFERENCE_CURVE_T_K = (
    770.440000, 788.823111, 806.894039, 824.656542, 842.114354, 859.271187, 876.130728, 892.696640, 908.972566, 924.962121,
    940.668899, 956.096472, 971.248385, 986.128162, 1000.739303, 1015.085285, 1029.169561, 1042.995561, 1056.566690, 1069.886332,
    1082.957845, 1095.784567, 1108.369809, 1120.716861, 1132.828987, 1144.709431, 1156.361411, 1167.788123, 1178.992738, 1189.978404,
    1200.748248, 1211.305369, 1221.652847, 1231.793737, 1241.731068, 1251.467850, 1261.007067, 1270.351679, 1279.504625, 1288.468817,
    1297.247148, 1305.842483, 1314.257667, 1322.495521, 1330.558840, 1338.450399, 1346.172948, 1353.729212, 1361.121896, 1368.353679,
    1375.427217, 1382.345144, 1389.110067, 1395.724575, 1402.191228, 1408.512566, 1414.691105, 1420.729337, 1426.629731, 1432.394731,
    1438.026761, 1443.528218, 1448.901478, 1454.148891, 1459.272787, 1464.275470, 1469.159221, 1473.926299, 1478.578936, 1483.119345,
    1487.549714, 1491.872205, 1496.088960, 1500.202096, 1504.213707, 1508.125863, 1511.940611, 1515.659975, 1519.285954, 1522.820525,
    1526.265642, 1529.623234, 1532.895207, 1536.083445, 1539.189806, 1542.216127, 1545.164220, 1548.035874, 1550.832855, 1553.556906,
    1556.209745, 1558.793067, 1561.308544, 1563.757825, 1566.142535, 1568.464275, 1570.724624, 1572.925137, 1575.067344, 1577.152754,
    1579.182850, 1581.159095, 1583.082925, 1584.955755, 1586.778976, 1588.553954, 1590.282033, 1591.964534, 1593.602754, 1595.197966,
    1596.751421, 1598.264344, 1599.737939, 1601.173387, 1602.571842, 1603.934439, 1605.262286, 1606.556469, 1607.818052, 1609.048073,
    1610.247547, 1611.417468, 1612.558804, 1613.672500, 1614.759479, 1615.820638, 1616.856853, 1617.868976, 1618.857835, 1619.824234,
    1620.768954, 1621.692755, 1622.596370, 1623.480509, 1624.345862, 1625.193092, 1626.022839, 1626.835721, 1627.632331, 1628.413241,
    1629.178997, 1629.930122, 1630.667117, 1631.390459, 1632.100600, 1632.797970, 1633.482976, 1634.156000, 1634.817402, 1635.467519,
    1636.106662, 1636.735120, 1637.353160, 1637.961024, 1638.558931, 1639.147075, 1639.725630, 1640.294743, 1640.854539, 1641.405121,
    1641.946566, 1642.478930, 1643.002243, 1643.516513, 1644.021726, 1644.517841, 1645.004797, 1645.482508, 1645.950865, 1646.409734,
    1646.858960, 1647.298363, 1647.727740, 1648.146864, 1648.555486, 1648.953333, 1649.340107, 1649.715489, 1650.079134, 1650.430676,
    1650.769724, 1651.095864, 1651.408659, 1651.707648, 1651.992346, 1652.262247, 1652.516819, 1652.755506, 1652.977732, 1653.182895,
    1653.370370, 1653.539508, 1653.689638, 1653.820065, 1653.930070, 1654.018911, 1654.085822, 1654.130015, 1654.150677, 1654.146972,
    1654.118042, 1654.063002, 1653.980948, 1653.870950, 1653.732054, 1653.563285, 1653.363641, 1653.132101, 1652.867617, 1652.569120,
    1652.235514, 1651.865685, 1651.458490, 1651.012766, 1650.527325, 1650.000958, 1649.432429, 1648.820481, 1648.163833, 1647.461180,
    1646.711194, 1645.912524, 1645.063795, 1644.163609, 1643.210543, 1642.203154, 1641.139971, 1640.019503, 1638.840235, 1637.600628,
    1636.299119, 1634.934122, 1633.504029, 1632.007206, 1630.441998, 1628.806725, 1627.099683, 1625.319147, 1623.463367, 1621.530570,
    1619.518958, 1617.426711, 1615.251986, 1612.992917, 1610.647611, 1608.214156, 1605.690614, 1603.075025, 1600.365404, 1597.559743,
    1602.954829, 1601.231423, 1599.507671, 1597.783588, 1596.059188, 1594.334486, 1592.609495, 1590.884232, 1589.158709, 1587.432941,
    1585.706943, 1583.980728, 1582.254310, 1580.527705, 1578.800925, 1577.073986, 1575.346901, 1573.619683, 1571.892348, 1570.164908,
    1568.437379, 1566.709773, 1564.982104, 1563.254387, 1561.526635, 1559.798862, 1558.071081, 1556.343306, 1554.615551, 1552.887829,
    1551.160154, 1549.432539, 1547.704998, 1545.977545, 1544.250191, 1542.522952, 1540.795840, 1539.068868, 1537.342051, 1535.615400,
    1533.888929, 1532.162652, 1530.436581, 1528.710730, 1526.985111, 1525.259738, 1523.534623, 1521.809780, 1520.085221, 1518.360959,
    1516.637008, 1514.913379, 1513.190086, 1511.467141, 1509.744556, 1508.022346, 1506.300522, 1504.579096, 1502.858082, 1501.137491,
    1499.417337, 1497.697632, 1495.978388, 1494.259617, 1492.541332, 1490.823545, 1489.106268, 1487.389514, 1485.673295, 1483.957622,
    1482.242509, 1480.527966, 1478.814007, 1477.100643, 1475.387885, 1473.675747, 1471.964240, 1470.253375, 1468.543165, 1466.833622,
    1465.124756, 1463.416581, 1461.709107, 1460.002346, 1458.296310, 1456.591010, 1454.886458, 1453.182665, 1451.479644, 1449.777404,
    1448.075959, 1446.375318, 1444.675494, 1442.976497, 1441.278340, 1439.581032, 1437.884586, 1436.189012, 1434.494322, 1432.800527,
    1431.107637, 1429.415664, 1427.724619, 1426.034512, 1424.345355, 1422.657158, 1420.969933, 1419.283689, 1417.598438, 1415.914191,
    1414.230957, 1412.548749, 1410.867576, 1409.187449, 1407.508378, 1405.830375, 1404.153449, 1402.477611, 1400.802871, 1399.129240,
    1397.456728, 1395.785345, 1394.115102, 1392.446009, 1390.778076, 1389.111313, 1387.445730, 1385.781337, 1384.118145, 1382.456163,
    1380.795401, 1379.135870, 1377.477579, 1375.820538, 1374.164757, 1372.510246, 1370.857013, 1369.205071, 1367.554427, 1365.905091,
    1364.257073, 1362.610384, 1360.965031, 1359.321025, 1357.678376, 1356.037092, 1354.397183, 1352.758658, 1351.121527, 1349.485799,
    1347.851484, 1346.218590, 1344.587127, 1342.957103, 1341.328529, 1339.701412, 1338.075763, 1336.451589, 1334.828901, 1333.207707,
    1331.588016, 1329.969836, 1328.353177, 1326.738048, 1325.124456, 1323.512412, 1321.901923, 1320.292998, 1318.685646, 1317.079876,
    1315.475696, 1313.873114, 1312.272138, 1310.672779, 1309.075043, 1307.478939, 1305.884476, 1304.291662, 1302.700504, 1301.111012,
    1299.523194, 1297.937057, 1296.352609, 1294.769860, 1293.188816, 1291.609487, 1290.031879, 1288.456001, 1286.881860, 1285.309465,
    1283.738824, 1282.169944, 1280.602832, 1279.037498, 1277.473948, 1275.912189, 1274.352231, 1272.794079, 1271.237743, 1269.683228,
    1268.130544, 1266.579696, 1265.030694, 1263.483543, 1261.938251, 1260.394827, 1258.853276, 1257.313606, 1255.775824, 1254.239938,
    1252.705954, 1251.173880, 1249.643723, 1248.115489, 1246.589186, 1245.064821, 1243.542400, 1242.021931, 1240.503420, 1238.986874,
    1237.472299, 1235.959704, 1234.449093, 1232.940474, 1231.433854, 1229.929239, 1228.426636, 1226.926050, 1225.427490, 1223.930960,
    1222.436468, 1220.944020, 1219.453621, 1217.965280, 1216.479001, 1214.994791, 1213.512656, 1212.032602, 1210.554636, 1209.078764,
    1207.604991, 1206.133324, 1204.663768, 1203.196330, 1201.731015, 1200.267829, 1198.806779, 1197.347869, 1195.891107, 1194.436496,
    1192.984044, 1191.533756, 1190.085637, 1188.639693, 1187.195929, 1185.754352, 1184.314966, 1182.877777, 1181.442790, 1180.010011,
    1178.579446, 1177.151099, 1175.724975, 1174.301080, 1172.879420, 1171.459998, 1170.042821, 1168.627894, 1167.215221, 1165.804808,
    1164.396659, 1162.990779, 1161.587174, 1160.185848, 1158.786807, 1157.390054, 1155.995595, 1154.603434, 1153.213576, 1151.826026,
    1150.440789, 1149.057868, 1147.677269, 1146.298996, 1144.923053, 1143.549446, 1142.178177, 1140.809253, 1139.442677, 1138.078453,
    1136.716585, 1135.357079, 1133.999937, 1132.645165, 1131.292766, 1129.942745, 1128.595105, 1127.249851, 1125.906986, 1124.566515,
    1123.228441, 1121.892769, 1120.559501, 1119.228643, 1117.900197, 1116.574167, 1115.250558, 1113.929372, 1112.610614, 1111.294286,
    1109.980394, 1108.668939, 1107.359926, 1106.053358, 1104.749238, 1103.447570, 1102.148357, 1100.851603, 1099.557310, 1098.265482,
    1096.976122, 1095.689234, 1094.404820, 1093.122884, 1091.843428, 1090.566456, 1089.291970, 1088.019974, 1086.750470, 1085.483462,
    1084.218951, 1082.956942, 1081.697437, 1080.440438, 1079.185948, 1077.933970, 1076.684506, 1075.437560, 1074.193133, 1072.951228,
    1071.711848, 1070.474995, 1069.240671, 1068.008880, 1066.779622, 1065.552901, 1064.328719, 1063.107078, 1061.887980, 1060.671428,
    1059.457423, 1058.245968, 1057.037064, 1055.830714, 1054.626920, 1053.425683, 1052.227006, 1051.030891, 1049.837338, 1048.646351,
    1047.457931, 1046.272079, 1045.088798, 1043.908088, 1042.729952, 1041.554392, 1040.381408, 1039.211002, 1038.043176, 1036.877932,
    1035.715270, 1034.555192, 1033.397699, 1032.242793, 1031.090475, 1029.940746, 1028.793608, 1027.649061, 1026.507107, 1025.367747,
    1024.230981, 1023.096812, 1021.965239, 1020.836264, 1019.709888, 1018.586111, 1017.464935, 1016.346360, 1015.230387, 1014.117017,
    1013.006250, 1011.898088, 1010.792530, 1009.689578, 1008.589232, 1007.491492, 1006.396359, 1005.303834, 1004.213916, 1003.126607,
    1002.041906, 1000.959814, 999.880332, 998.803458, 997.729195, 996.657541, 995.588497, 994.522063, 993.458239, 992.397025,
    991.338421, 990.282427, 989.229043, 988.178269, 987.130105, 986.084550, 985.041605, 984.001269, 982.963541, 981.928422,
    980.895910, 979.866007, 978.838710, 977.814020, 976.791937, 975.772459, 974.755586, 973.741318, 972.729654, 971.720593,
    970.714134, 969.710277, 968.709020, 967.710364, 966.714307, 965.720848, 964.729987, 963.741722, 962.756052, 961.772977,
    960.792495, 959.814605, 958.839306, 957.866596, 956.896476, 955.928942, 954.963995, 954.001632, 953.041853, 952.084656,
    951.130039, 950.178001, 949.228541, 948.281657, 947.337347, 946.395610, 945.456444, 944.519848, 943.585819, 942.654356,
    941.725458, 940.799122, 939.875346, 938.954129, 938.035468, 937.119363, 936.205810, 935.294808, 934.386354, 933.480447,
    932.577084, 931.676263, 930.777982, 929.882239, 928.989031, 928.098357, 927.210213, 926.324597, 925.441507, 924.560941,
    923.682895, 922.807368, 921.934356, 921.063858, 920.195870, 919.330390, 918.467416, 917.606943, 916.748970, 915.893494,
    915.040512, 914.190021, 913.342018, 912.496500, 911.653464, 910.812907, 909.974826, 909.139218, 908.306080, 907.475408,
    906.647199, 905.821451, 904.998160, 904.177321, 903.358933, 902.542992, 901.729494, 900.918436, 900.109814, 899.303625,
    898.499865, 897.698531, 896.899618, 896.103124, 895.309045, 894.517376, 893.728114, 892.941256, 892.156797, 891.374733,
    890.595061, 889.817776, 889.042875, 888.270353, 887.500207, 886.732432, 885.967024, 885.203980, 884.443294, 883.684962,
    882.928981, 882.175346, 881.424052, 880.675096, 879.928472, 879.184177, 878.442205, 877.702553, 876.965215, 876.230188,
    875.497465, 874.767044, 874.038919, 873.313084, 872.589537, 871.868271, 871.149282, 870.432565, 869.718115, 869.005927,
    868.295996, 867.588317, 866.882884, 866.179694, 865.478741, 864.780019, 864.083523, 863.389248, 862.697189, 862.007341,
    861.319698, 860.634254, 859.951005, 859.269944, 858.591067, 857.914367, 857.239840, 856.567480, 855.897280, 855.229236,
    854.563341, 853.899590, 853.237977, 852.578496, 851.921142, 851.265908, 850.612789, 849.961779, 849.312871, 848.666059,
    848.021338, 847.378702, 846.738144, 846.099658, 845.463237, 844.828877, 844.196569, 843.566309, 842.938089, 842.311904,
    841.687747, 841.065611, 840.445490, 839.827377, 839.211266, 838.597150, 837.985023, 837.374878, 836.766708, 836.160506,
    835.556266, 834.953981, 834.353644, 833.755247, 833.158786, 832.564251, 831.971636, 831.380935, 830.792140, 830.205244,
    829.620240, 829.037120, 828.455878, 827.876506, 827.298998, 826.723344, 826.149540, 825.577576, 825.007445, 824.439141,
    823.872655, 823.307980, 822.745109, 822.184034, 821.624746, 821.067240, 820.511506, 819.957537, 819.405326, 818.854864,
    818.306144, 817.759157, 817.213896, 816.670354, 816.128521, 815.588390, 815.049952, 814.513201, 813.978127, 813.444722,
    812.912979, 812.382888, 811.854443, 811.327633, 810.802452, 810.278890, 809.756939, 809.236591, 808.717838, 808.200670,
    807.685079, 807.171057, 806.658595, 806.147684, 805.638315, 805.130481, 804.624171, 804.119378, 803.616092, 803.114304,
    802.614006, 802.115189, 801.617843, 801.121960, 800.627530, 800.134545, 799.642995, 799.152871, 798.664164, 798.176865,
    797.690965, 797.206453, 796.723321, 796.241559, 795.761159, 795.282110, 794.804403, 794.328028, 793.852976, 793.379238,
    792.906803, 792.435663, 791.965806, 791.497225, 791.029908, 790.563846, 790.099029, 789.635447, 789.173091, 788.711950,
    788.252015, 787.793275, 787.335720, 786.879340, 786.424126, 785.970066, 785.517152, 785.065371, 784.614715, 784.165173,
    783.716734, 783.269389, 782.823126, 782.377936, 781.933807, 781.490730, 781.048693, 780.607687, 780.167700, 779.728722,
    779.290742, 778.853749, 778.417733, 777.982683, 777.548588, 777.115437, 776.683219, 776.251923, 775.821539, 775.392055,
    774.963460, 774.535743, 774.108893, 773.682899, 773.257749, 772.833433, 772.409938, 771.987255, 771.565371, 771.144275,
    770.723955, 770.304401, 769.885600, 769.467542, 769.050214, 768.633605, 768.217703, 767.802498, 767.387976, 766.974127,
    766.560938, 766.148397, 765.736494, 765.325215, 764.914550, 764.504486, 764.095011, 763.686113, 763.277780, 762.870000,
)



def equivalent_time_scale(equivalent_kg: float, reference_kg: float = REFERENCE_EQUIVALENT_KG) -> float:
    """相对标定当量的时间缩放：``(W / W_ref)^(2/3)``。"""
    w = float(equivalent_kg)
    if w <= 0:
        raise ValueError("当量必须大于 0")
    ref = float(reference_kg)
    if ref <= 0:
        raise ValueError("reference_kg 必须大于 0")
    return float((w / ref) ** (2.0 / 3.0))


def fireball_total_duration_s(equivalent_kg: float) -> float:
    """火球总持续时间 t_d ≈ 0.30·W^(1/3) (s)，与 engineering_tab 公式一致。"""
    w = float(equivalent_kg)
    if w <= 0:
        raise ValueError("当量 W 必须大于 0")
    return 0.30 * (w ** (1.0 / 3.0))


def fireball_total_duration_ms(equivalent_kg: float) -> float:
    return fireball_total_duration_s(equivalent_kg) * 1000.0


def embedded_reference_curve() -> tuple[np.ndarray, np.ndarray]:
    """内嵌参考温度曲线：时间 (ms) 与温度 (K)。"""
    t_ms = np.linspace(
        0.0, _REFERENCE_CURVE_T_SPAN_MS, _REFERENCE_CURVE_N, dtype=np.float64
    )
    t_k = np.asarray(_REFERENCE_CURVE_T_K, dtype=np.float64)
    if t_k.shape[0] != _REFERENCE_CURVE_N:
        raise ValueError("内嵌温度参考曲线长度与 _REFERENCE_CURVE_N 不一致")
    return t_ms, t_k


def reference_curve_duration_ms(csv_path: Optional[str | Path] = None) -> float:
    """参考温度曲线时间轴末点 (ms)。默认用内嵌数据；可传外部 CSV 覆盖。"""
    t_ms, _ = load_reference_curve(csv_path)
    return float(t_ms[-1])


def reference_baseline_simulation_duration_ms(
    equivalent_kg: float,
    csv_path: Optional[str | Path] = None,
) -> float:
    """与参考曲线一致的温度过程时长：``T_span × (W/100)^(2/3)``。"""
    span = reference_curve_duration_ms(csv_path)
    return span * equivalent_time_scale(equivalent_kg)


def default_temperature_curve_csv_path() -> Path:
    """历史 CSV 路径（可选覆盖源）；打包环境可不存在，默认走内嵌数据。"""
    return Path(__file__).resolve().parent / "fireball_temperature_reference_curve.csv"


def _load_reference_curve_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1))
    t_ms = np.asarray(d[:, 0], dtype=np.float64).ravel()
    t_k = np.asarray(d[:, 1], dtype=np.float64).ravel()
    if t_ms.shape[0] < 5:
        raise ValueError(f"温度参考曲线至少需要 5 个控制点：{path}")
    order = np.argsort(t_ms)
    return t_ms[order], t_k[order]


def load_reference_curve(
    csv_path: Optional[str | Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    加载参考温度曲线。

    - ``csv_path`` 为 None：使用内嵌数据。
    - 指向存在的文件：从该 CSV 读取。
    - 指向不存在的文件：告警后回退内嵌数据。
    """
    if csv_path is None:
        return embedded_reference_curve()
    path = Path(csv_path)
    if path.is_file():
        return _load_reference_curve_csv(path)
    warnings.warn(
        f"未找到温度参考 CSV：{path}，改用内嵌参考曲线。",
        UserWarning,
    )
    return embedded_reference_curve()


@dataclass
class UpwardCubic:
    coeffs: np.ndarray  # [p3, p2, p1, p0]，t 单位为 ms，T 为 °C

    def T(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        p = np.poly1d(self.coeffs)
        return p(t_ms)

    def dT(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        return np.polyder(np.poly1d(self.coeffs))(t_ms)


@dataclass
class DragDecay:
    A: float
    k: float
    t0: float
    T0: float

    def T(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(t_ms)
        return self.A + (self.T0 - self.A) * np.exp(-self.k * (t - self.t0))

    def dT(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(t_ms)
        return -(self.T0 - self.A) * self.k * np.exp(-self.k * (t - self.t0))


class FireballTemperatureCalculator:
    def __init__(
        self,
        blend_width_ms: float = 12.0,
        mode: Literal["blend", "c1"] = "blend",
        profile: Literal["reference_csv", "legacy"] = "reference_csv",
        reference_csv_path: Optional[str | Path] = None,
    ):
        """
        Args:
            blend_width_ms / mode: 仅在 ``profile='legacy'`` 时生效。
            profile: ``reference_csv``（默认用内嵌参考曲线）或 ``legacy``。
            reference_csv_path: 可选外部 CSV 覆盖；None 使用内嵌数据。
        """
        self.profile: Literal["reference_csv", "legacy"] = profile
        self.mode = mode
        self.blend_w = float(blend_width_ms)

        self.reference_curve_path: Optional[Path] = None
        self._interp_degC: Optional[PchipInterpolator] = None

        self.t_ms_all: Optional[np.ndarray] = None
        self.T_degC_all: Optional[np.ndarray] = None
        self.t0 = 35.0
        self.up_model: Optional[UpwardCubic] = None
        self.decay_model: Optional[DragDecay] = None
        self.p_coeffs: Optional[np.ndarray] = None

        self.t1 = 0.0
        self.t2 = 0.0
        self.t0_reference_peak_ms = float(self.t0)

        if profile == "legacy":
            self._init_legacy_piecewise(blend_width_ms)
        else:
            self._init_reference_pchip(reference_csv_path)

    def _init_reference_pchip(self, csv_path: Optional[str | Path]) -> None:
        t_ms, T_K = load_reference_curve(csv_path)
        T_degC = T_K - TEMP_OFFSET

        duplicates = np.zeros(t_ms.shape[0], dtype=bool)
        duplicates[1:] = t_ms[1:] == t_ms[:-1]
        kept = ~duplicates

        if csv_path is not None and Path(csv_path).is_file():
            self.reference_curve_path = Path(csv_path)
        else:
            self.reference_curve_path = None
        self.t_ms_reference = t_ms[kept]
        self.T_K_reference = T_K[kept]
        self._interp_degC = PchipInterpolator(
            self.t_ms_reference.astype(np.float64),
            (T_degC[kept]).astype(np.float64),
            extrapolate=True,
        )
        peak_i = int(np.argmax(self.T_K_reference))
        self.t0_reference_peak_ms = float(self.t_ms_reference[peak_i])

    def _init_legacy_piecewise(self, blend_width_ms: float) -> None:
        """原 6 点手工近似 + cubic / drag + blend."""
        self.t_ms_all = np.array([0, 20, 35, 70, 105, 140], dtype=float)
        self.T_degC_all = np.array([1180, 1240, 1220, 1015, 820, 740], dtype=float)
        self.t0 = 35.0
        self.t0_reference_peak_ms = float(self.t0)
        self.blend_w = float(blend_width_ms)
        self.t1 = max(0.0, self.t0 - self.blend_w / 2.0)
        self.t2 = self.t0 + self.blend_w / 2.0

        mask_up = self.t_ms_all <= self.t0
        self.p_coeffs = np.polyfit(
            self.t_ms_all[mask_up], self.T_degC_all[mask_up], deg=3
        )
        self.up_model = UpwardCubic(coeffs=self.p_coeffs)
        T0 = float(self.up_model.T(self.t0))
        dT0 = float(self.up_model.dT(self.t0))

        mask_decay = self.t_ms_all >= self.t0
        x = self.t_ms_all[mask_decay] - self.t0
        T_decay = self.T_degC_all[mask_decay]

        if self.mode == "c1":
            A_min = 500.0
            A_max = float(min(T_decay)) - 1.0
            A_grid = np.linspace(A_min, A_max, 401)
            best = None
            for A in A_grid:
                B = T0 - A
                if B <= 0:
                    continue
                k_c = -dT0 / B
                if k_c <= 0:
                    continue
                T_pred = A + (T0 - A) * np.exp(-k_c * x)
                sse = float(np.sum((T_decay - T_pred) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, A, k_c)
            if best is None:
                raise RuntimeError("C1 模式下未能找到有效的 A")
            _, A_opt, k_opt = best
        else:
            A_min = 500.0
            A_max = float(min(T_decay)) - 1.0
            k_min, k_max = 1e-3, 0.2
            A_grid = np.linspace(A_min, A_max, 401)
            k_grid = np.linspace(k_min, k_max, 400)
            best = None
            for A in A_grid:
                for k_c in k_grid:
                    T_pred = A + (T0 - A) * np.exp(-k_c * x)
                    sse = float(np.sum((T_decay - T_pred) ** 2))
                    if best is None or sse < best[0]:
                        best = (sse, A, k_c)
            assert best is not None
            _, A_opt, k_opt = best

        self.decay_model = DragDecay(A=A_opt, k=k_opt, t0=self.t0, T0=T0)

    def _blend_S_and_Sdot(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.clip((t - self.t1) / self.blend_w, 0.0, 1.0)
        S = 3 * s**2 - 2 * s**3
        dS_dt = (6 * s - 6 * s**2) / self.blend_w
        return S, dS_dt

    def temperature_modified(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        scalar = np.isscalar(t_ms)
        t = np.asarray(t_ms, dtype=float)
        if self.profile == "reference_csv" and self._interp_degC is not None:
            T_K = self._interp_degC(t) + TEMP_OFFSET
        else:
            assert self.up_model is not None and self.decay_model is not None
            if self.mode == "c1":
                T_C = np.where(
                    t <= self.t0,
                    self.up_model.T(t),
                    self.decay_model.T(t),
                )
            else:
                T_up = self.up_model.T(t)
                T_drag = self.decay_model.T(t)
                S, _ = self._blend_S_and_Sdot(t)
                T_C = (1 - S) * T_up + S * T_drag
            T_K = T_C + TEMP_OFFSET

        return float(T_K) if scalar else T_K

    def rate_modified(self, t_ms: np.ndarray | float) -> np.ndarray | float:
        scalar = np.isscalar(t_ms)
        t = np.asarray(t_ms, dtype=float)
        if self.profile == "reference_csv" and self._interp_degC is not None:
            deriv = self._interp_degC.derivative()(t)
            dT = np.asarray(deriv, dtype=float)
        else:
            assert self.up_model is not None and self.decay_model is not None
            if self.mode == "c1":
                dT = np.where(
                    t <= self.t0,
                    self.up_model.dT(t),
                    self.decay_model.dT(t),
                )
            else:
                Tup = self.up_model.T(t)
                Tdr = self.decay_model.T(t)
                dTup = self.up_model.dT(t)
                dTdr = self.decay_model.dT(t)
                S, dSdt = self._blend_S_and_Sdot(t)
                dT = (1 - S) * dTup + S * dTdr + dSdt * (Tdr - Tup)
        return float(dT) if scalar else dT

    def _reference_curve_span_ms(self) -> float:
        """CSV 基准时间跨度 (ms)。"""
        if (
            self.profile == "reference_csv"
            and getattr(self, "t_ms_reference", None) is not None
            and len(self.t_ms_reference) > 0
        ):
            return float(self.t_ms_reference[-1])
        return 140.0

    def _reference_peak_k_from_csv(self) -> float:
        """CSV 基准全局峰值温度 (K)，用于幅值缩放分母。"""
        if getattr(self, "T_K_reference", None) is not None and len(self.T_K_reference) > 0:
            return float(np.max(self.T_K_reference))
        if self.profile == "legacy" and self.T_degC_all is not None and len(self.T_degC_all) > 0:
            return float(np.max(self.T_degC_all)) + TEMP_OFFSET
        peak_ms = float(getattr(self, "t0_reference_peak_ms", self.t0))
        return float(self.temperature_modified(peak_ms))

    def _simulation_to_reference_time_ms(
        self,
        t_ms: Union[np.ndarray, float],
        duration_ms: float,
    ) -> np.ndarray:
        """
        物理时刻 → CSV 形状参数轴：``t_ref = (t / t_d) · T_csv,span``（``t_d = duration_ms``）。
        """
        t = np.asarray(t_ms, dtype=np.float64)
        dur = float(duration_ms)
        if dur <= 0:
            return np.zeros_like(t, dtype=np.float64)
        span = self._reference_curve_span_ms()
        return (t / dur) * span

    def temperature_baseline_scaled(
        self,
        t_ms: Union[np.ndarray, float],
        *,
        equivalent_kg: float,
        duration_ms: float,
        peak_temperature_k: float,
        ambient_k: float = 297.15,
        reference_equivalent_kg: float = REFERENCE_EQUIVALENT_KG,
    ) -> Union[np.ndarray, float]:
        """
        基准参考曲线时间缩放 + 峰值对齐。

        Args:
            t_ms: 仿真时刻 (ms)
            equivalent_kg: 保留兼容；时长与峰值由 ``duration_ms`` / ``peak_temperature_k`` 传入
            duration_ms: 火球温度过程时长 ``t_d`` (ms)，来自工程公式
            peak_temperature_k: 目标峰值 (K)，通常为 ``1.3·T_eq``（工程公式）
            ambient_k: 环境温度 (K)，幅值缩放基准
        """
        del equivalent_kg, reference_equivalent_kg
        scalar = np.isscalar(t_ms)
        t = np.asarray(t_ms, dtype=np.float64)
        t_ref = self._simulation_to_reference_time_ms(t, duration_ms)
        T_ref = np.asarray(self.temperature_modified(t_ref), dtype=np.float64)

        T_ref_peak = self._reference_peak_k_from_csv()
        denom = T_ref_peak - float(ambient_k)
        if denom <= 1e-6:
            T_out = np.full_like(t, float(peak_temperature_k), dtype=np.float64)
        else:
            factor = (float(peak_temperature_k) - float(ambient_k)) / denom
            T_out = float(ambient_k) + (T_ref - float(ambient_k)) * factor

        return float(T_out) if scalar else T_out

    def temperature_from_reference_shape(
        self,
        t_ms: Union[np.ndarray, float],
        duration_ms: float,
    ) -> Union[np.ndarray, float]:
        """
        仅按仿真时长拉伸 CSV 形状：``t_ref = (t/t_d)·T_csv,span``，不做工程峰值缩放。
        用于参数仿真（无当量/含铝输入）。
        """
        scalar = np.isscalar(t_ms)
        t = np.asarray(t_ms, dtype=np.float64)
        t_ref = self._simulation_to_reference_time_ms(t, duration_ms)
        T_K = np.asarray(self.temperature_modified(t_ref), dtype=np.float64)
        return float(T_K) if scalar else T_K

    def rate_baseline_scaled(
        self,
        t_ms: Union[np.ndarray, float],
        *,
        equivalent_kg: float,
        duration_ms: float,
        peak_temperature_k: float,
        ambient_k: float = 297.15,
        reference_equivalent_kg: float = REFERENCE_EQUIVALENT_KG,
    ) -> Union[np.ndarray, float]:
        """``temperature_baseline_scaled`` 对时间的导数 (K/ms)。"""
        del equivalent_kg, reference_equivalent_kg
        scalar = np.isscalar(t_ms)
        t = np.asarray(t_ms, dtype=np.float64)
        T_ref_peak = self._reference_peak_k_from_csv()
        denom = T_ref_peak - float(ambient_k)
        dur = float(duration_ms)
        span = self._reference_curve_span_ms()
        if denom <= 1e-6 or dur <= 0:
            dT = np.zeros_like(t, dtype=np.float64)
        else:
            factor = (float(peak_temperature_k) - float(ambient_k)) / denom
            t_ref = self._simulation_to_reference_time_ms(t, duration_ms)
            dT_ref = np.asarray(self.rate_modified(t_ref), dtype=np.float64)
            dT = factor * dT_ref * (span / dur)
        return float(dT) if scalar else dT

    def print_parameters(self) -> None:
        if self.profile == "reference_csv" and self._interp_degC is not None:
            print("Temperature profile: reference curve (PCHIP)")
            if self.reference_curve_path is not None:
                print(f"  source: {self.reference_curve_path}")
            else:
                print("  source: embedded reference curve")
            n = self.t_ms_reference.shape[0]
            print(f"  control points: {n}, t_span = [{self.t_ms_reference[0]:.3g}, {self.t_ms_reference[-1]:.3g}] ms")
            print(f"  peak ≈ {self.t0_reference_peak_ms:.3g} ms, T_peak ≈ {float(np.max(self.T_K_reference)):.3f} K")
            Ts = np.asarray(self.temperature_modified(self.t_ms_reference), dtype=float)
            rmse = float(np.sqrt(np.mean((Ts - self.T_K_reference) ** 2)))
            print(f"  RMSE(sampled control times, K): {rmse:g}")
            return

        assert self.p_coeffs is not None and self.decay_model is not None
        print("Temperature profile: legacy (6-point analytic)")
        p3, p2, p1, p0 = self.p_coeffs
        print("Upward polynomial (cubic) coefficients (t in ms, T in °C internally):")
        print(f"  p3 = {p3:.10e}")
        print(f"  p2 = {p2:.10e}")
        print(f"  p1 = {p1:.10e}")
        print(f"  p0 + offset = {p0 + TEMP_OFFSET:.10e} K")
        dm = self.decay_model
        print(f"  t0 = {self.t0:.1f} ms, T0 = {dm.T0 + TEMP_OFFSET:.6f} K")
        print(f"  mode = {self.mode}, blend width = {self.blend_w:.1f} ms")
        print("\nDrag decay (in K endpoints):")
        print(f"  A  = {dm.A + TEMP_OFFSET:.6f} K")
        print(f"  k  = {dm.k:.6f} (/ms)")
        assert self.T_degC_all is not None and self.t_ms_all is not None
        T_pred_K = self.temperature_modified(self.t_ms_all)
        T_true_K = self.T_degC_all + TEMP_OFFSET
        ss_res = float(np.sum((T_true_K - T_pred_K) ** 2))
        ss_tot = float(np.sum((T_true_K - np.mean(T_true_K)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"\nGlobal R^2 (legacy anchor points): {r2:.6f}")

    def plot(self, t_max_ms: Optional[float] = None, show_points: bool = True) -> None:
        if (
            self.profile == "reference_csv"
            and self._interp_degC is not None
            and t_max_ms is None
        ):
            t_max_ms = float(self.t_ms_reference[-1])
        elif t_max_ms is None:
            t_max_ms = 140.0

        n = max(600, int(t_max_ms) + 1)
        t = np.linspace(0.0, t_max_ms, n)
        T_mod_K = self.temperature_modified(t)
        dT_mod = self.rate_modified(t)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(
            t,
            T_mod_K,
            label=f"Temperature ({self.profile})",
            color="tab:orange",
            linewidth=2,
        )
        if show_points:
            if self.profile == "reference_csv" and self._interp_degC is not None:
                step = max(1, self.t_ms_reference.shape[0] // 42)
                idx = slice(None, None, step)
                plt.scatter(
                    self.t_ms_reference[idx],
                    self.T_K_reference[idx],
                    s=26,
                    label="CSV samples",
                    color="darkred",
                    zorder=4,
                    alpha=0.75,
                )
            elif self.T_degC_all is not None and self.t_ms_all is not None:
                plt.scatter(
                    self.t_ms_all,
                    self.T_degC_all + TEMP_OFFSET,
                    label="Digitized (legacy)",
                    marker="v",
                    color="tab:orange",
                )
        linestyle = "-" if self.profile == "legacy" else "--"
        xline = (
            float(self.t0)
            if self.profile == "legacy"
            else self.t0_reference_peak_ms
        )
        plt.axvline(xline, color="gray", linestyle=linestyle, linewidth=1, label="t_peak")
        plt.xlabel("t (ms)")
        plt.ylabel("T (K)")
        plt.title("Modified temperature vs time")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(t, dT_mod, label="dT/dt (K/ms)", color="tab:blue", linewidth=2)
        plt.axhline(0.0, color="gray", linewidth=1)
        plt.axvline(xline, color="gray", linestyle=linestyle, linewidth=1, label="t_peak")
        plt.xlabel("t (ms)")
        plt.ylabel("dT/dt (K/ms)")
        plt.title("Temperature rate vs time")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


def main() -> None:
    calc = FireballTemperatureCalculator()
    calc.print_parameters()
    demo_t = np.linspace(0, 140, 6)
    for tid in demo_t[:-1]:
        T = calc.temperature_modified(float(tid))
        dT = calc.rate_modified(float(tid))
        print(f"t = {tid:>6.2f} ms -> T = {T:8.3f} K, dT/dt = {dT:10.6f} K/ms")

    tt, KK = load_reference_curve()
    chk = tt <= 141
    preds = calc.temperature_modified(tt[chk])
    diff = preds - KK[chk]
    print(f"PCHIP vs embedded (t<=140 ms): max |Δ| = {float(np.max(np.abs(diff))):.4f} K")
    preds_all = calc.temperature_modified(tt)
    print(
        f"PCHIP vs embedded (full curve): RMSE(K)={float(np.sqrt(np.mean((preds_all - KK) ** 2))):.4g}; "
        f"max |Δ|={float(np.max(np.abs(preds_all - KK))):.4g} K（插值结点处≈数值零）"
    )
    print(
        "提示：在交互环境中可调用 calc.plot() 查看图形；"
        "无显示环境时请设置 MPLBACKEND=Agg 并在 plot() 中改用 savefig。"
    )

if __name__ == "__main__":
    main()
