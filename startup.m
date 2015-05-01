% Clean environment
clc; clear all; close all;

% Load Classification Toolbox
addpath(genpath('/opt/Classification_toolbox'));

% Load Neural Networks Toolbox
addpath(genpath('/opt/DeepLearnToolbox'));

% Load dataset
dataset = load('leaf.csv');