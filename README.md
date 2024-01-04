# Baseball Player Analysis: Aaron Judge

## Introduction

This project utilizes the `pybaseball` package to analyze data for the baseball player Aaron Judge. pybaseball is a Python package for baseball data analysis. This package scrapes Baseball Reference, Baseball Savant, and FanGraphs so you don't have to. The package retrieves statcast data, pitching stats, batting stats, division standings/team records, awards data, and more. Data is available at the individual pitch level, as well as aggregated at the season level and over custom time periods. See the docs for a comprehensive list of data acquisition functions. The analysis focuses on understanding Aaron Judge's performance and behavior within the strike zone using a Support Vector Machine (SVC). Since the strike zone of a batter is dependant on his size, the SVM can help to predict whether a strike or ball would be called based on pitch location.

## Requirements

- Python 3
- Install required packages: `pip install pybaseball scikit-learn matplotlib`

##PyBaseball

Installation: pip install pybaseball.<br>
This script analyzes data for baseball player Aaron Judge, and utilizes a Support Vector Classifier to analyze his strike zone. 
