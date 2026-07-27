import math
import os
from typing import List
from datetime import datetime

__all__ = ['TimeFuncs']

def lastest(timelines: List[dict], target_date = None) -> dict:
    temptimes = []
    tempstamps = []
    for timeline in timelines:
        temptime = timeline['TimeLine']
        temptimes.append(int(temptime))
        tempstamps.append(timeline)
    timestamp = tempstamps[temptimes.index(max(temptimes))]
    return timestamp

def earliest(timelines: List[dict], target_date = None) -> dict:
    temptimes = []
    tempstamps = []
    for timeline in timelines:
        temptime = timeline['TimeLine']
        temptimes.append(int(temptime))
        tempstamps.append(timeline)

    timestamp = tempstamps[temptimes.index(min(temptimes))]
    return timestamp



def nearest(timelines: List[dict], target_date: str) -> dict:
    temptimes = []
    tempstamps = []
    for timeline in timelines:
        temp_date = timeline['TimeLine']
        target_ = datetime.strptime(target_date, '%Y%m')
        temp_ = datetime.strptime(temp_date, '%Y%m')
        delta = target_ - temp_
        distance = abs(delta.days // 30)
        temptimes.append(distance)
        tempstamps.append(timeline)

    timestamp = tempstamps[temptimes.index(min(temptimes))]
    return timestamp

def green(timelines: List[dict], target_date = None) -> dict:
    temptimes = []
    tempstamps = []
    for timeline in timelines:
        temptime = timeline['TimeLine']
        month = int(temptime[4:])
        if 4<=month<=10:
            temptimes.append(int(temptime))
            tempstamps.append(timeline)

    if not temptimes:
        return []
    else:
        timestamp = tempstamps[temptimes.index(max(temptimes))]
        return timestamp
    
def spring(timelines: List[dict], target_date = None) -> dict:
    temptimes = []
    tempstamps = []
    for timeline in timelines:
        temptime = timeline['TimeLine']
        month = int(temptime[4:])
        if 3<=month<=5:
            temptimes.append(int(temptime))
            tempstamps.append(timeline)
    if not temptimes:
        return []
    else:
        timestamp = tempstamps[temptimes.index(max(temptimes))]
        return timestamp

def summer(timelines: List[dict], target_date = None) -> dict:
    temptimes = []
    tempstamps = []
    for timeline in timelines:
        temptime = timeline['TimeLine']
        month = int(temptime[4:])
        if 6<=month<=8:
            temptimes.append(int(temptime))
            tempstamps.append(timeline)
    if not temptimes:
        return []
    else:
        timestamp = tempstamps[temptimes.index(max(temptimes))]
        return timestamp
    
def autumn(timelines: List[dict], target_date = None) -> dict:
    temptimes = []
    tempstamps = []
    for timeline in timelines:
        temptime = timeline['TimeLine']
        month = int(temptime[4:])
        if 9<=month<=11:
            temptimes.append(int(temptime))
            tempstamps.append(timeline)
    if not temptimes:
        return []
    else:
        timestamp = tempstamps[temptimes.index(max(temptimes))]
        return timestamp


def winter(timelines: List[dict], target_date = None) -> dict:
    temptimes = []
    tempstamps = []
    for timeline in timelines:
        temptime = timeline['TimeLine']
        month = int(temptime[4:])
        if month>=12 or month<=2:
            temptimes.append(int(temptime))
            tempstamps.append(timeline)
    if not temptimes:
        return []
    else:
        timestamp = tempstamps[temptimes.index(max(temptimes))]
        return timestamp


def year(timelines: List[dict], year: str) -> dict:
    # temptimes = []
    tempstamps = []
    for timeline in timelines:
        temp_date = timeline['TimeLine']
        target_ = year
        temp_ = temp_date[:4]
        # 如果处于该年份内
        if temp_ == target_:
            tempstamps.append(timeline)
    if not tempstamps:
        return []
    timestamp = tempstamps[0]
    return timestamp

class TimeFuncs:
    Latest   = lastest
    Earliest = earliest
    Green    = green
    Year     = year
    Nearest  = nearest
    Spring   = spring
    Summer   = summer
    Autumn   = autumn
    Winter   = winter
