from scipy.stats import t,norm

x = [1,2,3,4,5,6,7,8,9]
# Mean: 5
# Median: 5
# Standard deviation: 2.738612788
# Variance: 6.666666666667
y = [178501.22217228636,154526.5922952518,133107.62349543348]
# Mean: 148224.648261
# Median: 146850.07680475712
# Standard deviation: 4756.78245039
# Variance: 22626979.3
z = [1.5,2.7,4.3,6.9,10.8]
# Mean: 5.24
# Median: 5.24
# Standard deviation:3.31638357251
# Variance: 10.9984
#w = range(0,65,15)
w = range(1,10000,1)


data = w

t_parameters = t.fit(data)

print('Parameter[0]: ' + str(t_parameters[0]))
print('Parameter[1]: ' + str(t_parameters[1]))
print('Parameter[2]: ' + str(t_parameters[2]))

print('Mean: ' + str(t.mean(df=t_parameters[0], loc=t_parameters[1], scale=t_parameters[2])))
print('Median: ' + str(t.median(df=t_parameters[0], loc=t_parameters[1], scale=t_parameters[2])))
print('Standard deviation:' + str(t.std(df=t_parameters[0], loc=t_parameters[1], scale=t_parameters[2])))
print('Variance: ' + str(t.var(df=t_parameters[0], loc=t_parameters[1], scale=t_parameters[2])))



norm_parameters = norm.fit(data)

print('Parameter[0]: ' + str(norm_parameters[0]))
print('Parameter[1]: ' + str(norm_parameters[1]))

print('Mean: ' + str(norm.mean(loc=norm_parameters[0], scale=norm_parameters[1])))
print('Median: ' + str(norm.median(loc=norm_parameters[0], scale=norm_parameters[1])))
print('Standard deviation:' + str(norm.std(loc=norm_parameters[0], scale=norm_parameters[1])))
print('Variance: ' + str(norm.var(loc=norm_parameters[0], scale=norm_parameters[1])))