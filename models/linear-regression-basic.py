from sklearn.linear_model import LinearRegression

reg = LinearRegression()

x = [[1],[2],[3],[4],[5],[6]]
y = [[2],[4],[6],[7],[9],[14]]

reg.fit(x,y)

reg.predict([[100]])
