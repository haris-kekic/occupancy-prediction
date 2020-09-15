# -----------------------------------------------------------
# Helper Klasse für Datummanipulation
# (C) 2020 Haris Kekic, Fernuniversität in Hagen
# -----------------------------------------------------------

import datetime 
import pandas as pd

def get_weekday(date, lang="de"):
    ''' Gibt den Namen des Wochetages in ensprechender Sprache zurück lang='de' oder lang='en' '''
    if (lang == 'de'):
        days=["Montag","Dienstag","Mittwoch","Donnerstag","Freitag","Samstag","Sonntag"]
    else:
        days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    return days[date.weekday()]

def get_weekday_index(date):
    ''' Wandelt den Tag in eine Nummer wobei 0=Sonntag, 1=Montag ... 6=Samstag '''
    if (date.weekday() < 6):
        weekday_number = date.weekday() + 1
    else:
        weekday_number = 0 #Sonntag

    return weekday_number

def is_weekend(date):
    ''' Gibt zurück ob das Datum Wocheende ist oder nicht '''
    index = get_weekday_index(date)
    return True if index == 5 or index == 6 else False

def is_workday(date):
    ''' Gibt zurück ob das Datum Werktag ist oder nicht '''
    return not is_weekend(date)

def get_season(date):
    seasons = { 'WINTER': 0,  'SPRING': 1, 'SUMMER': 2, 'AUTUMN': 3 }
    month = date.month
    day = date.day
    season = seasons['WINTER']
    if month in (1, 2, 3):
        season = seasons['WINTER']
    if (month == 3 and day > 19):
        season = seasons['SPRING']
    elif month in (4, 5, 6):
        season = seasons['SPRING']
        if (month == 6 and day > 20):
            season = seasons['SUMMER']
    elif month in (7, 8, 9):
        season = seasons['SUMMER']
        if (month == 9 and day > 21):
            season = seasons['AUTUMN']
    elif month in (10, 11, 12):
        season = seasons['AUTUMN']
        if (month == 12 and day > 20):
            season = seasons['WINTER']

    return season


    