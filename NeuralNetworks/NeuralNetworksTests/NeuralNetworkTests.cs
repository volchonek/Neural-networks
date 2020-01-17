using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetworks.Tests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var outputs = new double[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            var inputs = new double[,]
            {
                /*
                 * Результат - пацент болен -1
                 *             пациент здоров -0
                 *             
                 * Неправильная температура T
                 * Хороший возраст A
                 * Курит S
                 * Правильно питается F
                */
                //  T A S F
                  { 0,0,0,0}, // 0
                  { 0,0,0,1}, // 1
                  { 0,0,1,0}, // 2
                  { 0,0,1,1}, // 3
                  { 0,1,0,0}, // 4
                  { 0,1,0,1}, // 5
                  { 0,1,1,0}, // 6
                  { 0,1,1,1}, // 7
                  { 1,0,0,0}, // 8
                  { 1,0,0,1}, // 9
                  { 1,0,1,0}, // 10
                  { 1,0,1,1}, // 11
                  { 1,1,0,0}, // 12
                  { 1,1,0,1}, // 13
                  { 1,1,1,0}, // 14
                  { 1,1,1,1}  // 15
            };

            var topology = new Topology(4, 1, 0.01, 2);
            var neuralNetwork = new NeuralNetwork(topology);

            var difference = neuralNetwork.Learn(outputs, inputs, 400000); // средняя квадратичная ошибка при обучении

            var results = new List<double>();

            for (int i = 0; i < outputs.Length; i++)
            {
                var row = NeuralNetwork.GetRow(inputs, i);
                var res = neuralNetwork.FeedForward(row).Output;
                results.Add(res);
            }

            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 3);
                var actual = Math.Round(results[i], 3);

                Assert.AreEqual(expected, actual);
            }
        }

        [TestMethod]
        public void DatasetTest()
        {
            var outputs = new List<double>();
            var inputs = new List<double[]>();

            using (var sr = new StreamReader("../../../hearts.csv"))
            {
                var header = sr.ReadLine();

                while (!sr.EndOfStream)
                {
                    var row = sr.ReadLine();
                    var values = row.Split(',').Select(value => Convert.ToDouble(value.Replace(".", ","))).ToList();
                    var output = values.Last();
                    var input = values.Take(values.Count - 1).ToArray();

                    outputs.Add(output);
                    inputs.Add(input);
                }
            }

            var inputSignals = new double[inputs.Count, inputs[0].Length];
            for (int i = 0; i < inputSignals.GetLength(0); i++)
            {
                for (var j = 0; j < inputSignals.GetLength(1); j++)
                {
                    inputSignals[i, j] = inputs[i][j];
                }
            }

            var topology = new Topology(outputs.Count, 1, 0.1, outputs.Count / 2);
            var neuralNetwork = new NeuralNetwork(topology);
            var difference = neuralNetwork.Learn(outputs.ToArray(), inputSignals, 100); // средняя квадратичная ошибка при обучении

            var results = new List<double>();
            for (int i = 0; i < outputs.Count; i++)
            {
                var res = neuralNetwork.FeedForward(inputs[i]).Output;
                results.Add(res);
            }

            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 3);
                var actual = Math.Round(results[i], 3);

                //Assert.AreEqual(expected, actual);
            }
        }
    }
}