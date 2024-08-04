import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# buat dummy data
bedrooms = np.array([1,2,3,2,2,1,4,2,4,6,3,5,5,5,6,3])
house_price = np.array([10000, 10000, 14000, 15000, 20000, 22000, 28000, 30000, 30000,40000, 44000,48000,58000,50000,60000,18000])

#latih model dengan linear regression.fit()
bedrooms = bedrooms.reshape(-1,1)
linreg = LinearRegression()
linreg.fit(bedrooms, house_price)

# menampilkan plot hubungan antara jarak kamar dengan harga rumah
plt.scatter(bedrooms, house_price)
plt.plot(bedrooms, linreg.predict(bedrooms))
plt.show()

print(len(bedrooms))
print(len(house_price))