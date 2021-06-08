# 20190828 BeginningPython Chapter 3
# Working with Strings

# 20190828 page 52
# String Method

# Center
print("Monday, Tuesday, Wednesday".center(39))
print("Thursday, Friday Saturday".center(39, "*"))

# Find
print("Monday, Tuesday, Wednesday".find('day'))
day = "Thursday, Friday Saturday"
print(day.find('Thursday'))
print(day.find('Friday'))
print(day.find('Saturday'))
print(day.find('Sunday'))

# 20190831 Join
dirs = '', 'usr', 'bin', 'env'
'/'.join(dirs)
print('C:' + '\\'.join(dirs))

# Lower
print('Thursday, Friday, Saturday'.lower())
if 'Friday' in (day): print('Found The Friday!')
elif 'friday' in (day): print('Found The friday!')
else: print('There was no Friday or friday!')

# 20190902 Title Casing page 55
friday_search = ('   There was no Friday or friday!   ')
print(friday_search)
print('Now in title format')
print((friday_search).title())
print('Turning day into night')
print(friday_search.replace('day', 'night'))
print('Time to split the title.')
print((friday_search).split())
print('Time to get rid of all that whitespace')
print((friday_search).strip())

# 20190903 Translation page 57
table = str.maketrans('cs', 'kz')
print('This is the table')
print(table)
print('this is an incredible test'.translate(table))