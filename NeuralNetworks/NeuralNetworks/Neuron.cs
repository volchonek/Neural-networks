using System;
using System.Collections.Generic;

namespace NeuralNetworks
{
    class Neuron
    {
        public List<double> Weights { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; set; }

        public Neuron(int inputCount, NeuronType neuronType = NeuronType.Normal)
        {
            NeuronType = neuronType;
            Weights = new List<double>();

            // set weights = 1
            for (int i = 0; i < inputCount; i++)
            {
                Weights.Add(1);
            }
        }

        // TODO: добавить проверку на входные данные
        public double FeedForward(List<double> inputs)
        {
            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum = inputs[i] * Weights[i];
            }

            Output = Sigmoid(sum);

            return Output;
        }

        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Exp(-x));

            return result;
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    
        public void SetWeight(params double[] weights)
        {
            // TO DO: удалить после добавления возможности обучения сети
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }
    }
}
