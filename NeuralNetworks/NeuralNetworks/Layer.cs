using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks
{
    class Layer
    {
        // TO DO: подумать о переносе свойсвта "NeuronType" в слой
        public List<Neuron> Neurons { get; }
        public int Count => Neurons?.Count ?? 0;

        public Layer(List<Neuron> neurons, NeuronType neuronType = NeuronType.Normal)
        {
            // TO DO: проверить все входные нейроны на соответсвие типу 
            Neurons = neurons;
        }

        // take all output signals in current layer
        public List<double> GetSignals()
        {
            var result = new List<double>();

            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }

            return result;
        }
    }
}
