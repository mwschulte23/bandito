# place for getting into the details
# 1. analyzing events -> what bandit would look like for this subset
# 2. suggested bandit splits based on segments
#   ^^ creates some feature improvements on bandit id -> arm -> retrieval/steps (e.g this forces user to choose which id to pull from)
#   ^^ will need to think about this; an extensive rebuild to have bandit variants which I don't like at all...maybe a bandit family?
#   ^^ a bandit family is interesting, enables a single api call IF segment info is presented.
#   ^^ this would be an bigger update (e.g bandit A needs events moved to bandit B w/ both in same "family")