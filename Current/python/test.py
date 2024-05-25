from sklearn.metrics import (explained_variance_score, r2_score, mean_squared_error, mean_absolute_error)

y_true = [7.89813519, 7.74595269, 7.7843827, 7.64142302, 7.20485754, 7.22484189,
          7.23560224, 7.13414646, 7.25865931, 7.0296168, 7.19870859, 7.0219315,
          6.9957982, 6.98503785, 7.17565153, 7.12031222, 7.13722152, 7.42621475,
          7.58762008, 7.80436588, 7.65064585, 7.50307476, 7.51998406, 7.55380265,
          7.40469404, 7.33705685, 7.16796506, 7.33859438, 7.34781721, 7.00809609,
          6.85283853, 6.90971536, 7.20024612, 7.43390005, 7.63527408, 7.9027466,
          7.73826622, 7.84279588, 8.09489664, 7.63066267, 7.59069513, 7.31553614,
          7.44004899, 7.93810272, 7.79514305, 7.93195378, 7.839722, 7.89967272,
          7.90120907, 7.72289445, 7.73672869, 7.79667941, 7.82742412, 7.78284517,
          7.59223266, 7.57685972, 7.67062903, 8.15023594, 7.99344203, 8.05339274,
          8.19481489, 8.21326171, 8.39311387, 8.50379247, 8.80354606, 8.54529636,
          8.55913178, 8.59756178, 8.62369391, 8.57296602, 8.64675215, 8.86042289,
          8.91576219, 9.01106786, 8.84197606, 8.70362781, 8.45306458, 8.44998953,
          8.48995706, 8.6313792, 9.0894654, 9.57983267, 9.80580012, 9.62441044,
          9.38921899, 9.39229405, 9.56138584, 9.81348659, 9.66898939, 9.66284045,
          9.41842617, 9.11867258, 9.36462439, 9.53525372, 9.72894011, 9.5383276,
          9.51988195, 9.44148441, 9.28930191, 9.4199637, 9.25240826, 9.21244189,
          9.26163226, 9.37230969, 9.10022576, 8.496106, 8.33162562, 8.0380198,
          7.9749952, 8.03494592, 8.04416875, 7.7751587, 7.71674551, 7.58454619,
          7.72135692, 7.93810272, 7.78591905, 7.70752268, 7.80590341, 7.87507695,
          7.94271414, 7.76439834, 7.91350696, 7.96730873, 7.99344203, 7.89659766,
          8.13332664, 8.15023594, 8.11180593, 8.13486417, 8.1102684, 7.72904339,
          7.78130764, 7.82742412, 7.71828304, 7.68907586, 7.61221584, 7.60145549,
          7.7029101, 7.45849582, 7.47233005, 7.51537265, 7.40008263, 7.42006581,
          7.43236369, 7.31246226, 7.30477579, 7.11723717, 7.0849561, 6.99887326,
          6.9573682, 6.94353396, 7.1572047, 6.96351714, 7.04191468, 7.20793259,
          7.21100648, 7.19870859, 7.38778474, 7.57378584, 7.53689336, 7.11108822,
          6.49928317, 6.28253737, 6.13342875, 6.23642206, 6.62840743, 7.1817993,
          7.36626403, 7.39239616, 7.34320579, 7.19870859, 7.03269185, 7.06189786,
          6.79442534]
y_pred = [7.782819, 7.8435245, 7.799773, 7.7657657, 7.511659, 7.2052326,
          7.2048063, 7.2954516, 7.016838, 7.2705984, 7.18334, 7.185836,
          7.03871, 7.0023966, 7.04834, 7.241689, 7.20566, 7.2873,
          7.5010753, 7.668284, 7.8353634, 7.569871, 7.5756283, 7.601527,
          7.408959, 7.4133368, 7.3051744, 7.2443705, 7.4382105, 7.196971,
          6.9850316, 6.853385, 7.082753, 7.266213, 7.574535, 7.8084655,
          7.9791975, 7.7897787, 7.93264, 7.927863, 7.0424647, 7.7811294,
          7.2193327, 7.6241493, 7.893051, 7.938421, 7.9111547, 7.8695655,
          7.8014235, 7.929623, 7.738844, 7.799783, 7.8262596, 7.7184205,
          7.71583, 7.6939464, 7.5501785, 7.755196, 8.125784, 8.051094,
          8.15992, 8.168257, 8.296391, 8.457461, 8.589364, 8.712068,
          8.577096, 8.524124, 8.316754, 8.718743, 8.570974, 8.771047,
          8.849587, 8.956869, 8.969028, 8.75107, 8.64404, 8.423762,
          8.491971, 8.606665, 8.599049, 9.231513, 9.434183, 9.880053,
          9.539004, 9.358075, 9.412803, 9.616262, 9.775864, 9.610101,
          9.527099, 9.379895, 9.09742, 9.410387, 9.450175, 9.553429,
          9.654288, 9.55004, 9.389347, 9.302585, 9.402568, 9.216683,
          9.211814, 9.24849, 9.216164, 8.488101, 8.700952, 8.094249,
          8.05381, 7.8535123, 8.123327, 7.999897, 7.723854, 7.668899,
          7.571517, 7.707752, 8.023735, 7.7497907, 7.6601744, 7.817126,
          7.8478847, 7.9028172, 7.790579, 7.9460697, 7.961236, 7.8639064,
          7.918906, 8.184504, 8.127489, 8.177984, 8.082628, 8.081533,
          7.6412444, 7.835175, 7.7502127, 7.6911736, 7.7246895, 7.5583935,
          7.673543, 7.495253, 7.5519466, 7.445915, 7.506775, 7.411531,
          7.3980646, 7.4063497, 7.1665387, 7.321782, 7.0475073, 7.154747,
          7.011676, 7.0489864, 7.043817, 7.2186966, 7.098021, 7.1321306,
          7.234744, 7.2054276, 7.2625556, 7.4347634, 7.528234, 7.5648613,
          7.0286207, 6.6152496, 6.5083904, 6.2472606, 6.337406, 6.895822,
          7.369319, 7.4354362, 7.436028, 7.3687124, 7.094543, 6.996103,
          6.9449253]
print(explained_variance_score(y_true, y_pred))
print(r2_score(y_true, y_pred))
print(mean_squared_error(y_true, y_pred))
print(mean_absolute_error(y_true, y_pred))
