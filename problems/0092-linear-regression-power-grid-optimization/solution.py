import math

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	n = len(consumption_data)

	detrended_data = []
	for i in range(1, n+1):
		fluctuation = 10 * math.sin((2 * PI * i)/10)
		clean_value = consumption_data[i - 1] - fluctuation
		detrended_data.append(clean_value)

	x = list(range(1, n + 1))
	y = detrended_data

	sum_x = sum(x)
	sum_y = sum(y)
	sum_xy = sum([x[i] * y[i] for i in range(n)])
	sum_x2 = sum([xi ** 2 for xi in x])

	m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
	c = (sum_y - m * sum_x) / n


	day = 15
	b_predict = (m * day) + c

	fl_15 = 10 * math.sin((2 * PI * day)/ 10)
	f_predict = fl_15 + b_predict

	res = f_predict * 1.05

	return math.ceil(res)