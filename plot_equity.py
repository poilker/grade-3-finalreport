import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("equity_curve.csv")
df["time"] = pd.to_datetime(df["time"], utc=True)

plt.figure()
plt.plot(df["time"], df["equity"])
plt.title("Equity Curve")
plt.xlabel("Time")
plt.ylabel("Equity")
plt.tight_layout()

plt.savefig("equity_curve.png", dpi=200)
plt.show()
