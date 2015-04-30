% Clean environment
clc; clear all; close all;

% Load Neural Networks Toolbox
addpath(genpath('/opt/DeepLearnToolbox'));

% Load dataset
dataset = load('leaf.csv');