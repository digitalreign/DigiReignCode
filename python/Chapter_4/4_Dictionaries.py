# 20190904 BeginningPython Chapter 4
# Dictionaries: When Indices Won’t Do

# 20190904 page 59
# Dictionary Uses

months = ['January', 'February', 'March', 'April', 'May']
numbers = ['31', '28', '31', '30', '31']
print(numbers[months.index('February')])

# 20190905 page 60
# Creating and Using Dictionaries

calender = [('month', 'January'), ('days', 31)]
cal = dict(calender)
print(cal)

# Basic Dictionary Operations page 61
monthly = {}
monthly[1] = 'January'
monthly[2] = 'February'
monthly[3] = 'March'
monthly[4] = 'April'
monthly[5] = 'May'
monthly[6] = 'June'
monthly[7] = 'July'
monthly[8] = 'August'
monthly[9] = 'September'
monthly[10] = 'October'
monthly[11] = 'November'
monthly[12] = 'December'
print('The Monthly Dictionary')
print(monthly)
# 20190905 page 61
print('This should print February:')
print(monthly[2])
