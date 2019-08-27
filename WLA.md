

# 加权LASSO函数_基于坐标下降法
class WeigthedLasso(object):
    def __init__(self, lamrange, iteramax=1e2, tol=1e-5, intercept=True, selectmethod = 'AIC', **kwargs):
        super(WeigthedLasso,  **kwargs)
        self.lamrange = lamrange
        self.iteramax = iteramax
        self.tol = tol
        self.intercept = intercept
        self.weight_history = []
        self.selectmethod = selectmethod

    def weightmatrix(self, x, y, weight, params):
        try:
            weight.shape[1]
            residuals = y - np.dot(x, params)
            return float(np.dot(residuals.T, weight * residuals))
        except:
            print('please check input dim of weight matrix')
            return 0


    def lossfunction(self, x, y, weight, params, lamvalue):
        try:
            x.shape[1], y.shape[1], params.shape[1]
            return self.weightmatrix(x, y, weight, params) + lamvalue*abs(params).sum()
        except:
            print('check dim')
            print('xshape', x.shape)
            print('yshape', y.shape)
            print('params\' shape', params.shape)
            return 0

    def basefit(self,x, y, weight, lamvalue):
        m, n = x.shape
        if self.intercept:
            n += 1
            x = np.c_[np.ones((m, 1)), x]
        self.params = np.random.normal(0,1,(n,1))
        it, lastresult = 0, self.lossfunction(x, y, weight, self.params, lamvalue)
        while it < self.iteramax:
            if it//100:
                print('#', end='')
            it += 1
            # 坐标下降法求解， 避免梯度下降在0处没有梯度。
            for k in range(n):
                # 计算常量值z_k和p_k
                z_k = (x[:, k] @ (weight * x[:, k]).T)[0]
                p_k = 0
                for i in range(m):
                    # 梯度加权
                    p_k += weight[i,0] * x[i, k] * (y[i, 0] - sum([x[i, j] * self.params[j, 0] for j in range(n) if j != k]))
                if p_k < -lamvalue / 2:
                    w_k = (p_k + lamvalue / 2) / z_k
                elif p_k > lamvalue / 2:
                    w_k = (p_k - lamvalue / 2) / z_k
                else:
                    w_k = 0
                self.params[k, 0] = w_k
            loss = self.lossfunction(x, y, weight, self.params, lamvalue)
            delta = lastresult - loss
            if abs(delta) < self.tol:
                print('iteration', it)
                print('delta', delta)
                break


    def mse(self,y, ypre):
        return np.dot((y-ypre).T, (y-ypre))

    def AIC(self, y, ypre, length_data):
        return len(np.nonzero(self.params.flatten())[0]) * 2 + length_data * np.log(self.mse(y, ypre) / length_data)


    def fit(self, x, y, weight, type='notCV'):
        self.result = []
        self._pathparmas = []
        if self.selectmethod == 'AIC':
            selfunc = partial(self.AIC, length_data=x.shape[0])
        elif self.selectmethod == 'MSE':
            selfunc = self.mse

        if type == 'notCV':
            for lam in self.lamrange:
                self.basefit(x, y, weight, lam)
                self._pathparmas.append(self.params)
                self.result.append(selfunc(y.flatten(), self.predict(x)))
            self.result = np.array(self.result)
            try:
                self.best_lambda = self.lamrange[np.asscalar(np.where(self.result == np.min(self.result))[0])]
                self.params = self._pathparmas[np.asscalar(np.where(self.result == np.min(self.result))[0])]
            except:
                self.best_lambda = self.lamrange[0]
                self.params = self._pathparmas[0]
        elif type == 'CV':
            pass
        else:
            print('wrong select type')

    def predict(self, xtest):
        try:
            xtest.shape[1]
            if self.intercept:
                xtest = np.c_[np.ones((xtest.shape[0], 1)), xtest]
            return np.dot(xtest, self.params).flatten()
        except:
            print('check XTEST shape')
            print('xtest shape is ', xtest.shape, 'while the params shape is ', self.params.shape)

from sklearn.linear_model import ElasticNetCV
# 加权ElasticNet函数_基于坐标下降法
class WeightedElasnet(WeigthedLasso):
    def __init__(self, alpharange, rhorange, iteramax=1e3, tol=1e-5, intercept=True, selectmethod = 'AIC', paramtype='alpha',**kwargs):
        super().__init__(self, **kwargs)
        self.alpharange = alpharange
        self.rhorange = rhorange
        self.iteramax = iteramax
        self.tol = tol
        self.intercept = intercept
        self.weight_history = []
        self.selectmethod = selectmethod
        self.paramtype = paramtype

    @staticmethod
    def penalty_change(a, b, changetype='to_penalty'):
        if changetype == 'to_penalty':
            l1, l2 = a * b, a * (1 - b) / 2
            return l1, l2
        elif changetype == 'to_alpha':
            alpha, rho = 2 * b + a,  a /(2 * b + a)
            return alpha, rho

    def lossfunction(self, x, y, weight, params, penlist):
        a, b = penlist[0], penlist[1]
        try:
            non_used = x.shape[1], y.shape[1], params.shape[1]# 快速鉴别输入格式
            return super().weightmatrix(x, y, weight, params) + a * abs(params).sum() + b * (params**2).sum()
        except:
            print('check dim')
            print('xshape', x.shape)
            print('yshape', y.shape)
            print('params\' shape', params.shape)
            return 0

    def basefit(self,x, y, weight, penlist):
        if self.paramtype == 'alpha':
            a, b = penlist[0], penlist[1]
            self.l1, self.l2 = self.penalty_change(a ,b, 'to_penalty')
        else:
            self.l1, self.l2 = penlist[0], penlist[1]
            a, b = self.penalty_change(penlist[0], penlist[1], 'to_alpha')
        m, n = x.shape
        if self.intercept:
            n += 1
            x = np.c_[np.ones((m,1)), x]
        self.params = np.random.normal(0,1,(n,1))
        it, lastresult = 0, self.lossfunction(x, y, weight, self.params, (a, b))
        while it < self.iteramax:
            if it//100:
                print('#', end='')
            it += 1
            # 坐标下降法求解， 避免梯度下降在0处没有梯度。
            for k in range(n):
                # 计算常量值z_k和p_k
                z_k = (x[:, k].T @ (weight*x[:, k]))[0]
                p_k = 0
                for i in range(m):
                    # 坐标加权
                    p_k += weight[i, 0]*x[i, k] * (y[i, 0] - sum([x[i, j] * self.params[j, 0] for j in range(n) if j != k]))
                if p_k < -self.l1 / 2:
                    w_k = (p_k + self.l1 / 2) / (z_k + self.l2)
                elif p_k > self.l1 / 2:
                    w_k = (p_k - self.l1 / 2) / (z_k + self.l2)
                else:
                    w_k = 0
                self.params[k, 0] = w_k
            loss = self.lossfunction(x, y, weight, self.params, (a, b))
            delta = lastresult - loss
            if abs(delta) < self.tol:
                print('iteration', it)
                print('delta', delta)
                break
        print('')

    def fit(self, x, y, weight, type='notCV'):
        result = []
        self._pathparmas = []
        if self.selectmethod == 'AIC':
            selfunc = partial(self.AIC, length_data=x.shape[0])
        elif self.selectmethod == 'MSE':
            selfunc = self.mse

        if type == 'notCV':
            for alpha in self.alpharange:
                for rho in self.rhorange:
                    self.basefit(x, y, weight, (alpha, rho))
                    self._pathparmas.append(self.params)
                    result.append(selfunc(y.flatten(), self.predict(x)))
            result = np.array(result)
            try:
                d = np.asscalar(np.where(result == np.min(result))[0])
            except:
                d = 0
            self.best_penalty = (self.alpharange[d//len(self.rhorange)], self.rhorange[d % len(self.rhorange)])
            self.params = self._pathparmas[d]
        elif type == 'CV':
            pass
        else:
            print('wrong select type')
