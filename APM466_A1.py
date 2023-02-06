import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy import optimize
import datetime
import math
from matplotlib.pyplot import plot, xlabel, ylabel, title

# data importing and filtering
df = pd.read_excel('APM466_A1.xlsx')
df['maturity_date'] = pd.to_datetime(df['Maturity Date'])
df['issue_date'] = pd.to_datetime(df['Issue Date'])
data = df.iloc[:, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]


# bond calculations
class Bond:
    def __init__(self, ISIN, coupon_rate, quoted_price, maturity_date,
                 maturity_terms):
        self.ISIN = ISIN
        self.coupon_rate = coupon_rate
        self.quoted_price = quoted_price
        self.maturity_date = maturity_date
        # in units of year
        self.maturity_terms = maturity_terms

    def calculate_ytm(self, bond_price, ttm, accrued_days, face_value=100,
                      freq=2, x0=0.05):
        coupon = face_value * self.coupon_rate / freq
        # collection of coupon periods in years
        if self.maturity_terms < 0.5:
            terms = [ttm[t] for t in range(int(self.maturity_terms * 2))]
        else:
            terms = [self.maturity_terms]
            num = self.maturity_terms
            num = num - 0.5
            while num > 0:
                terms.append(num)
                num = num - 0.5
        # periods of coupons & face are transferred to units of months
        get_y = lambda ytm: sum([coupon / (1 + ytm / freq) **
                                 (freq * f) for f in terms]) + face_value / (
                                    1 + ytm
                                    / freq) ** (freq * self.maturity_terms) - \
                            (bond_price + accrued_days * face_value
                             * self.coupon_rate)
        return optimize.newton(get_y, x0)

    def calculate_spot_rate(self, index, ttm, spot_rate_list,
                            freq=2, face_value=100):
        coupon = (face_value * self.coupon_rate) / freq
        price = self.quoted_price[index]
        notional = face_value + coupon
        # self.maturity_terms need to start with a bond w/ maturity date
        # less than 0.5 year
        if self.maturity_terms < 0.5:
            return -(np.log(price/ notional) / self.maturity_terms)
        else:
            for k in range(len(spot_rate_list)):
                price -= coupon * np.exp(-spot_rate_list[k] * ttm[k])
            return -(np.log(price / notional) / self.maturity_terms)


bonds_list = []
# assume the date start at 2023-02-01, a year has 365 days
record_date = pd.Timestamp(2023, 2, 1)
time_to_maturity = []
for i in range(10):
    time_to_maturity.append((data.iloc[i, 12] - record_date).days / 365)
# print(time_to_maturity)
for i in range(10):
    bonds_list.append(Bond(ISIN=data.iloc[i, 0],
                           coupon_rate=(data.iloc[i, 1]),
                           quoted_price=data.iloc[i, 2:12],
                           maturity_date=data.iloc[i, 13],
                           maturity_terms=time_to_maturity[i]))

# yield to maturity
time = ['1-16', '1-17', '1-18', '1-19', '1-20', '1-23', '1-24', '1-25', '1-26',
        '1-27']
yield_curve = {}
accrued_terms = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
# ai_days_list = []
for i in range(10):
    yield_curve[time[i]] = []
    for j in range(10):
        if time_to_maturity[j] < accrued_terms[j]:
            ai_days = accrued_terms[j] - time_to_maturity[j]
            value = bonds_list[j].calculate_ytm(bond_price=bonds_list[j].quoted_price[i],
                                                ttm=time_to_maturity,
                                                accrued_days=ai_days)
        else:
            ai_days = accrued_terms[j + 1] - time_to_maturity[j]
            value = bonds_list[j].calculate_ytm(bond_price=bonds_list[j].quoted_price[i],
                                                ttm=time_to_maturity,
                                                accrued_days=ai_days)
        yield_curve[time[i]].append(value)

# for k in range(10):
#     if time_to_maturity[k] < accrued_terms[k]:
#         ai_days = accrued_terms[k] - time_to_maturity[k]
#         ai_days_list.append(ai_days)
#     else:
#         ai_days = accrued_terms[k + 1] - time_to_maturity[k]
#         ai_days_list.append(ai_days)

# print(time_to_maturity)
# print(ai_days_list)

# plot setting
plt.figure(figsize=(10, 6))
plt.xlabel('time to maturity')
plt.ylabel('yield to maturity')
plt.title('Five Year Yield Curve')
yield_x_axis = ['23/8', '24/3', '24/9', '25/3', '25/9', '26/3', '26/9',
                '27/3', '27/9', '28/3']

for date in yield_curve:
    plt.plot(yield_x_axis, yield_curve[date])
plt.legend(time, loc="upper right")

plt.savefig('yield_curve.png')

# spot curve
spot_curve = {}
plt.figure(figsize=(10, 6))
# plt.legend(time, loc="lower right")
plt.xlabel('time to maturity')
plt.ylabel('zero coupon yield')
plt.title('Five Year Spot Curve')
spot_x_axis = []
for i in range(10):
    spot_x_axis.append(round(time_to_maturity[i], 1))
for i in range(10):
    spot_curve[time[i]] = []
    spot_rate_l = []
    for j in range(10):
        value = bonds_list[j].calculate_spot_rate(index=i, ttm=time_to_maturity,
                                      spot_rate_list=spot_rate_l)
        spot_curve[time[i]].append(value)
        spot_rate_l.append(value)

for date in spot_curve:
    plt.plot(spot_x_axis, spot_curve[date])
plt.legend(time, loc="upper right")

plt.savefig('spot_curve.png')

# forward curve
forward_yield_curve = {}
plt.figure(figsize=(10, 6))
plt.xlabel('time to maturity')
plt.ylabel('forward yield')
plt.title('Five Year Forward Yield')
forward_axis = time_to_maturity[3:]

one_year_forward = []
for date in spot_curve:
    one_year_forward.append(spot_curve[date][1])
time_forward = time_to_maturity[3:]
# print(time_forward)
for i in range(10):
    forward_yield_curve[time[i]] = []
    spot_list = spot_curve[time[i]][3:]
    for j in range(len(time_forward)):
        value = (spot_list[j] * time_forward[j] -
                 one_year_forward[i]) / (time_forward[j] - 1)
        forward_yield_curve[time[i]].append(value)

for date in forward_yield_curve:
    plt.plot(forward_axis, forward_yield_curve[date])
plt.legend(time, loc="upper right")
plt.savefig('forward_yield_curve.png')

# covariance matrix
yield_matrix = np.zeros([5, 9])
forward_matrix = np.zeros([4, 9])
# print(yield_matrix)
# print(forward_matrix)
for i in range(5):
    for j in range(9):
        yield_matrix[i, j] = np.log(yield_curve[time[j+1]][2*i+1] /
                                    yield_curve[time[j]][2*i+1])

for i in range(4):
    for j in range(9):
        forward_matrix[i, j] = np.log(forward_yield_curve[time[j+1]][2*i]
                                     / forward_yield_curve[time[j]][2*i])

# print(yield_matrix)
# print(forward_matrix)
cov_yield = np.cov(yield_matrix)
cov_forward = np.cov(forward_matrix)
# print(cov_yield)
# print(cov_forward)
eigenvalue_yield = np.linalg.eig(cov_yield)[0]
eigenvector_yield = np.linalg.eig(cov_yield)[1]
eigenvalue_forward = np.linalg.eig(cov_forward)[0]
eigenvector_forward = np.linalg.eig(cov_forward)[1]
# print(eigenvalue_yield)
# print(eigenvector_yield)
# print(eigenvalue_forward)
# print(eigenvector_forward)
perc_yield = []
perc_forward = []
sum_yield = 0
sum_forward = 0
for i in eigenvalue_yield:
    sum_yield += i
for j in eigenvalue_yield:
    perc_yield.append(j/sum_yield)
for i in eigenvalue_forward:
        sum_forward += i
for j in eigenvalue_forward:
    perc_forward.append(j/sum_forward)

print(perc_yield)
print(perc_forward)
