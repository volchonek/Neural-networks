using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks
{
    public class Topology
    {
        // TO DO: подумать на обобщение  слоев в топологии 
        public int InputCount { get; }
        public int OutputCount { get; }
        public double LearningRate { get; }
        public List<int> HiddenLayers { get; }

        public Topology(int inputCount, int outputCount, double learningRate, params int[] layers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            LearningRate = learningRate;
            HiddenLayers = new List<int>();
            HiddenLayers.AddRange(layers);
        }
    }
}
