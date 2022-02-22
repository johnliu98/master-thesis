from src.systems import Pendulum
from src.bayesian_optimization import BangBang, SafeBangBang
from src.mpc import SafetyFilter

sys = Pendulum()
filter = SafetyFilter(sys, 25)
model = BangBang(sys, filter)

p = model.learn(iterations=300)
model.plot()
# p = [8, 48]

X, U = model.run(p)
sys.initialize_figure()
for x, u in zip(X.T, U.reshape(1, -1).T):
    sys.animate(x, u)
