using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks
{
    class Topology
    {
        // TO DO: подумать на обобщение  слоев в топологии 
        public int InputCount { get; }
        public int OutputCount { get; }
        public List<int> HiddenLayers { get; }

        public Topology(int inputCount, int outputCount, params int[] layers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;

            HiddenLayers = new List<int>();
            HiddenLayers.AddRange(layers);
        }
    }
}
