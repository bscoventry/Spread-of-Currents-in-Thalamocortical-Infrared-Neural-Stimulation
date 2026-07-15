import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
df1 = pd.read_pickle('CSDWave1.pkl')
df2 = pd.read_pickle('CSDWave2.pkl')
df3 = pd.read_pickle('CSDWave3.pkl')

frames = [df1, df2, df3]
df = pd.concat(frames, ignore_index=True)

numRows = len(df)

colNames = ['AnimalId','EPP','ISI','numSink']
numSink = pd.DataFrame(columns=colNames,dtype=object )
colNames = ['AnimalId','EPP','ISI','numSour']
numSour = pd.DataFrame(columns=colNames,dtype=object )
colNames = ['AnimalId','EPP','ISI','velSink']
velocitySink = pd.DataFrame(columns=colNames,dtype=object )
colNames = ['AnimalId','EPP','ISI','velSour']
velocitySour = pd.DataFrame(columns=colNames,dtype=object )
colNames = ['AnimalId','EPP','ISI','angleSink']
angSink = pd.DataFrame(columns=colNames,dtype=object )
colNames = ['AnimalId','EPP','ISI','angleSour']
angSour = pd.DataFrame(columns=colNames,dtype=object )
colNames = ['AnimalId','EPP','ISI','distSink']
distSink = pd.DataFrame(columns=colNames,dtype=object )
colNames = ['AnimalId','EPP','ISI','distSour']
distSour = pd.DataFrame(columns=colNames,dtype=object )
colNames = ['AnimalId','EPP','ISI','latSink']
latSink = pd.DataFrame(columns=colNames,dtype=object )
colNames = ['AnimalId','EPP','ISI','latSour']
latSour = pd.DataFrame(columns=colNames,dtype=object )
colNames = ['AnimalId','EPP','ISI','durSink']
durSink = pd.DataFrame(columns=colNames,dtype=object )
colNames = ['AnimalId','EPP','ISI','durSour']
durSour = pd.DataFrame(columns=colNames,dtype=object )

for ck in range(numRows):
    curAID = df.DataID[ck]
    curEPP = df.Energy[ck]
    curISI = df.ISI[ck]

    curNumSink = df.numSinkWaves[ck]
    row = pd.Series([curAID,curEPP,curISI,curNumSink],index=numSink.columns)
    numSink.loc[len(numSink)] = row

    curNumSour = df.numSourceWaves[ck]
    row = pd.Series([curAID,curEPP,curISI,curNumSour],index=numSour.columns)
    numSour.loc[len(numSour)] = row

    curSinkLat = df.latencySink[ck]
    if curSinkLat:
        row = pd.Series([curAID,curEPP,curISI,curSinkLat[0]],index=latSink.columns)
        latSink.loc[len(latSink)] = row
    
    curSourLat = df.latencySource[ck]
    if curSourLat:
        row = pd.Series([curAID,curEPP,curISI,curSourLat[0]],index=latSour.columns)
        latSour.loc[len(latSour)] = row
    
    curVelSink = df.velocitySink[ck]
    if len(curVelSink)>0:
        num = len(curVelSink)
        for bc in range(num):
            row = pd.Series([curAID,curEPP,curISI,curVelSink[bc]],index=velocitySink.columns)
            velocitySink.loc[len(velocitySink)] = row
    
    curVelSour = df.velocitySource[ck]
    if len(curVelSour)>0:
        num = len(curVelSour)
        for bc in range(num):
            row = pd.Series([curAID,curEPP,curISI,curVelSour[bc]],index=velocitySour.columns)
            velocitySour.loc[len(velocitySour)] = row
    
    curAngSink = df.angleSink[ck]
    if len(curAngSink)>0:
        num = len(curAngSink)
        for bc in range(num):
            row = pd.Series([curAID,curEPP,curISI,curAngSink[bc]],index=angSink.columns)
            angSink.loc[len(angSink)] = row
    
    curAngSour = df.angleSource[ck]
    if len(curAngSour)>0:
        num = len(curAngSour)
        for bc in range(num):
            row = pd.Series([curAID,curEPP,curISI,curAngSour[bc]],index=angSour.columns)
            angSour.loc[len(angSour)] = row
    
    curDistSink = df.distanceSink[ck]
    if len(curDistSink)>0:
        num = len(curDistSink)
        for bc in range(num):
            row = pd.Series([curAID,curEPP,curISI,curDistSink[bc]],index=distSink.columns)
            distSink.loc[len(distSink)] = row
    
    curDistSour = df.distanceSource[ck]
    if len(curDistSour)>0:
        num = len(curDistSour)
        for bc in range(num):
            row = pd.Series([curAID,curEPP,curISI,curDistSour[bc]],index=distSour.columns)
            distSour.loc[len(distSour)] = row

    curDurSink = df.durationSink[ck]
    if len(curDurSink)>0:
        num = len(curDurSink)
        for bc in range(num):
            row = pd.Series([curAID,curEPP,curISI,curDurSink[bc]],index=durSink.columns)
            durSink.loc[len(durSink)] = row
    
    curDurSour = df.durationSource[ck]
    if len(curDurSour)>0:
        num = len(curDurSour)
        for bc in range(num):
            row = pd.Series([curAID,curEPP,curISI,curDurSour[bc]],index=durSour.columns)
            durSour.loc[len(durSour)] = row

numSink.to_pickle('numSink.pkl')
numSour.to_pickle('numSour.pkl')

durSink.to_pickle('durSink.pkl')
durSour.to_pickle('durSour.pkl')

distSink.to_pickle('distSink.pkl')
distSour.to_pickle('distSour.pkl')

velocitySink.to_pickle('velocitySink.pkl')
velocitySour.to_pickle('velocitySour.pkl')

angSink.to_pickle('angSink.pkl')
angSour.to_pickle('angSour.pkl')

latSink.to_pickle('latSink.pkl')
latSour.to_pickle('latSour.pkl')

pdb.set_trace()