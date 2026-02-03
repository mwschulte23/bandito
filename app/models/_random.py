from sqlmodel import Session, create_engine

engine = create_engine("sqlite:///bandits.db")
SQLModel.metadata.create_all(engine)

# 1. Initialize the experiment
new_bandit = BanditState.create_new("drift_monitor_v1", dimensions=17, lambda_reg=1.0)

# 2. Save it
with Session(engine) as session:
    session.add(new_bandit)
    session.commit()