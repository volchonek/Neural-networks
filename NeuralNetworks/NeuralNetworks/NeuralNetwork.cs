using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworks
{
    class NeuralNetwork
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
        public Neuron FeedForward(List<double> inputSignals)
        {
            // TO DO: проверить совпадение количество входных сигналов должно совпадать с количестовм выходных нейронов во входном слое
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

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 0; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayeSignalsr = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayeSignalsr);
                }
            }
        }

        private void SendSignalsToInputNeurons(List<double> inputSignals)
        {
            for (int i = 0; i < inputSignals.Count; i++)
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
                    var neuron = new Neuron(lastLayer.Count, NeuronType.Normal);
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
                var neuron = new Neuron(lastLayer.Count, NeuronType.Normal);
                outputsNeorons.Add(neuron);
            }

            var outputLayer = new Layer(outputsNeorons, NeuronType.Output);
            Layers.Add(outputLayer);
        } 
    }
}
