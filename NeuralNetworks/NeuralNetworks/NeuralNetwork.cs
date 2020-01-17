using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworks
{
    public class NeuralNetwork
    {
        public Topology Topology { get; set; }
        public List<Layer> Layers { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayer();
            CreateOutputLayer();
        }

        // FeedForward для работы нейросети
        public Neuron FeedForward(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }
        }

        public double Learn(double[] expected, double[,] inputs, int epoch)
        {
            var signals = Normalization(inputs);

            var error = 0.0;

            for (int i = 0; i < epoch; i++)
            {             
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(signals, j);

                    error += BackPropagation(output, input);

                }
            }

            var result = error / epoch;

            return result;
        }

        public static double[] GetRow(double[,] matrix, int row)
        {
            var columns = matrix.GetLength(1);
            var array = new double[columns];

            for (int i = 0; i < columns; i++)
            {
                array[i] = matrix[row,i];
            }
            return array;
        }

        // масштабирование
        private double[,] Scalling(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                var min = inputs[0, column];
                var max = inputs[0, column];

                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    var item = inputs[row, column];

                    if (item < min)
                    {
                        min = item;
                    }

                    if (item > max)
                    {
                        max = item;
                    }
                }

                var devider = max - min;

                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - min) / devider;
                }
            }

            return result;
        }

        // нормализация
        private double[,] Normalization(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                var sum = 0.0;

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row,column];
                }

                var average = sum / inputs.GetLength(0);
                var error = 0.0;

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    error += Math.Pow((inputs[row,column] - average),2);
                }

                var standartError = Math.Sqrt(error/inputs.GetLength(0));

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - average) / standartError;
                }
            }

            return result;
        }

        // TODO: улучшить expexted для нескольких нейронов
        // метод обратного распостранения ошибки
        private double BackPropagation(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;

            // вычисляем дельту для последнего (выходного) и предпоследнего слоев
            var difference = actual - expected;

            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];

                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];

                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];

                        // вычисляем ошибку для остальных слоев
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;

                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }

            var result = difference * difference;

            return result;
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSignals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        private void CreateInputLayer()
        {
            var inputsNeorons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Normal);
                inputsNeorons.Add(neuron);
            }

            var inputLayer = new Layer(inputsNeorons, NeuronType.Input);
            Layers.Add(inputLayer);
        }

        private void CreateHiddenLayer()
        {
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var hiddenNeorons = new List<Neuron>();
                var lastLayer = Layers.Last();

                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Normal);
                    hiddenNeorons.Add(neuron);
                }

                var hiddenLayer = new Layer(hiddenNeorons, NeuronType.Normal);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateOutputLayer()
        {
            var outputsNeorons = new List<Neuron>();
            var lastLayer = Layers.Last();

            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Normal);
                outputsNeorons.Add(neuron);
            }

            var outputLayer = new Layer(outputsNeorons, NeuronType.Output);
            Layers.Add(outputLayer);
        } 
    }
}
