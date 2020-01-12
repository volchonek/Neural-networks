using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks
{
    public class Layer
    {
        // TO DO: подумать о переносе свойста "NeuronType" в слой
        public List<Neuron> Neurons { get; }
        public int NeuronCount => Neurons?.Count ?? 0;
        public NeuronType Type;

        public Layer(List<Neuron> neurons, NeuronType neuronType = NeuronType.Normal)
        {
            // TO DO: проверить все входные нейроны на соответсвие типу 
            Neurons = neurons;
            Type = neuronType;
        }

        // берем все выходные сигналы текущего слоя
        public List<double> GetSignals()
        {
            var result = new List<double>();

            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }

            return result;
        }

        public override string ToString()
        {
            return Type.ToString();
        }
    }
}
