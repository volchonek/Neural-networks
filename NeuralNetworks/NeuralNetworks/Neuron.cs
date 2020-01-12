using System;
using System.Collections.Generic;

namespace NeuralNetworks
{
    public class Neuron
    {
        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; set; }
        public double Delta { get; private set; }

        public Neuron(int inputCount, NeuronType neuronType = NeuronType.Normal)
        {
            NeuronType = neuronType;
            Weights = new List<double>();
            Inputs = new List<double>();

            // установка случайных весов
            InitWeightsRandomValue(inputCount);
        }

        private void InitWeightsRandomValue(int inputCount)
        {
            var rnd = new Random();

            for (int i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(rnd.NextDouble());
                }
           
                Inputs.Add(0);
            }
        }

        // TODO: добавить проверку на входные данные
        public double FeedForward(List<double> inputs)
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;

            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else 
            {
                Output = sum;
            }

            return Output;
        }

        // функция активации
        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Exp(-x));

            return result;
        }

        // обратная функция активации
        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid / 1.0 * (1.0 - sigmoid);

            return result;
        }

        public void Learn(double error, double learningRate)
        {
            if (NeuronType == NeuronType.Input)
            {
                return;
            }

            // вычисление дельты
            Delta = error * SigmoidDx(Output);

            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                // вычисление нового веса
                var newWeight = weight - input * Delta * learningRate;
                Weights[i] = newWeight;
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }

        // ручная установка весов ( для сброса ), используется если сеть попала в воронку
        public void SetWeights(params double[] weights)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }    
    }
}
