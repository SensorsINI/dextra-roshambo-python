import os
from typing import Union
import matplotlib.pyplot as plt
import csv
from glob import glob
from numpy import genfromtxt
import numpy as np
import matplotlib.ticker as plticker
# from matplotlib_scalebar.scalebar import ScaleBar
import datetime
import time

dir=os.path.join('logging/actions') # can't import globals_and_utils.py since it uses libs not always available

def plot_data():
    data=np.empty((0,6),dtype=int) # year,day_of_year,weekday,hour,minute,museum_movements_this_hour
    elapsed_minutes=[] # populated with correct data depending on CSV format (old with only 6 columns or new with elapsed minutes column)
    minutes=[]
    for f in glob(os.path.join(dir,'*.csv')):
        if os.path.getsize(f)==0:
            print(f'{f} is empty, skipping')
            continue   
        print(f'file {f}:')
        try:
            rows = genfromtxt(f, delimiter=',', encoding="utf8", dtype=int, skip_header=1)
        except Exception as e:
            print(f'error reading {f}: {e}')
            continue
        # print(row)
        if len(rows.shape)==2 and rows.shape[0]>1: # has more than 1 row of data (first row is header that is int-parsed as vector of -1 values)
            if rows.shape[1]==6: # old format, does not have minutes_since_last
                data=np.append(data,rows[:,:],axis=0)
                elapsed_minutes.append(data)
            elif rows.shape[1]==7: # new format
                idx=[0,1,2,3,4,6]
                data=np.append(data,rows[:,idx],axis=0)
                
    years=data[:,0]
    days=data[:,1]
    hours=data[:,3]
    minutes=data[:,4]
    moves=data[:,5]

    datetimes=[] # absolute datetimes from column data
    for i in range(len(years)):
        str_datetime=f'{years[i]:04d}-{days[i]:03d} {hours[i]:02d}:{minutes[i]:02d}'
        datetimes.append(datetime.datetime.strptime(str_datetime,'%Y-%j %H:%M'))
    
    # sort all the data in ascending time
    sort_idx=np.argsort(datetimes)
    datetimes=[datetimes[i] for i in sort_idx]
    
    moves=np.array([moves[i] for i in sort_idx])
    years=np.array([years[i] for i in sort_idx])
    days=np.array([days[i] for i in sort_idx])
    hours=np.array([hours[i] for i in sort_idx])
    minutes=np.array([minutes[i] for i in sort_idx])
    
    # get absolute times in hours since epoch 1970
    datetimes_hours=np.zeros(len(datetimes))
    for i in range(len(datetimes_hours)):
        datetimes_hours[i]=datetimes[i].timestamp()/(60*60)
    delta_times_hours=np.diff(datetimes_hours,prepend=datetimes_hours[0])

    moves_per_hour=moves/delta_times_hours
    
    year_frac_days=(days+hours/24.0+minutes/(24.*60)) # fraction of year in days
    frac_days=year_frac_days%7 # fraction of week
    int_weeks=np.floor(year_frac_days/7) # floored year weeks
    start_week=np.min(int_weeks)
    moves_nonzero=moves.astype(np.float32)
    moves_nonzero[moves_nonzero==0]=np.nan
    max_moves=np.max(moves)
    norm_moves=(moves_nonzero/max_moves)+int_weeks # shift weeks vertically
    
    # start/end date
    date_start= datetimes[0]
    date_end  = datetimes[-1]
    date_now=datetime.datetime.now()

    fig=plt.figure('Dextra activity', figsize=(10,4))
    fig.clear()
    x=frac_days
    y=norm_moves
    # plt.plot(x,y,'o',ms=3)
    stair_edges=np.append(x,x[-1])
    plt.stairs(values=y,edges=stair_edges,orientation='vertical',fill=False, baseline=int_weeks)
    plt.xlabel('days of week')
    plt.ylabel('week of year')
    plt.xlim([0,7])
    plt.title(f'Dextra robot hand movements (max {max_moves})')
    xticks=plt.gca().get_xticks()
    weekdays=['Su','M','Tu','W','Th','Fr','Sa','']
    # with_noons=[]
    # for w in weekdays:
    #     with_noons.append(w)
    #     with_noons.append('12')
    # plt.xticks(range(len(with_noons)),with_noons)
    plt.xticks(range(len(weekdays)),weekdays)
    loc = plticker.MultipleLocator(base=1.0)
    plt.gca().yaxis.set_major_locator(loc)
    plt.grid(True)
    # for w in int_weeks:
    #     plt.plot([0,8],[w,w],linewidth=0.05,color='k')

    total_moves=np.sum(moves)
    total_days=np.max(year_frac_days)-np.min(year_frac_days)
    moves_per_day=total_moves/total_days
    t=f'{date_start.strftime("%Y-%b-%d")} to {date_end.strftime("%Y-%b-%d")}\n{total_days:.1f}d: {total_moves:,} movements ({moves_per_day:.1f} moves/d)\nGenerated {date_now.strftime("%a %Y-%b-%d %H:%M")}'
    plt.text(.1,np.min(int_weeks)+.1,t, color='b', fontsize=12)
    print(t)
    
    day_of_week_now=date_now.hour/24+(date_now.weekday()+8)%7 # sunday is zero
    week=int(date_now.strftime("%U"))

    plt.plot([day_of_week_now,day_of_week_now],[week,week+1],'r')
    plt.draw()

plt.ion()

while True:
    plot_data()
    plt.pause(3600)
