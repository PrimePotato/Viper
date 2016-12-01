
import pandas as pd
import numpy as np

from functions import get_days_between


class Event(object):
    def __init__(self, ts, duration, weight):
        self.ts = ts
        self.duration = duration
        self.weight = weight


class EventList(object):
    def __init__(self, events):
        self.events = events
        self.eds = [e.ts for e in self.events]

    def calc_quote_bend(self, ts, start, end):
        pre_dys = get_days_between(start, ts)
        due_dys = get_days_between(start, end)
        due = [e for e in self.events if start <= e.ts <= end]
        pre = [e for e in self.events if start <= e.ts < ts]
        pre_sum = sum([d.weight*d.duration for d in pre])
        due_sum = sum([d.weight * d.duration for d in due])
        q_dys = ((pre_dys + pre_sum) / (due_dys + due_sum), pre_dys / due_dys)
        return np.sqrt(q_dys[0] / q_dys[1]), q_dys

ev1 = Event(pd.Timestamp('1/1/2011'), 0.3, 5)
ev2 = Event(pd.Timestamp('2/10/2011'), 0.3, 50)
ev3 = Event(pd.Timestamp('3/10/2011'), 0.3, 5)

el = EventList([ev1, ev2, ev3])
x = el.calc_quote_bend( pd.Timestamp('2/12/2011'), pd.Timestamp('12/12/2010'), pd.Timestamp('3/3/2011'))
print(x)




