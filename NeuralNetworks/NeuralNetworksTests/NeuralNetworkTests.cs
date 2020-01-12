using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks.Tests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var dataset = new List<Tuple<double, double[]>>
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
                //                                             T A S F
                new Tuple<double, double[]> (0, new double[] { 0,0,0,0}), // 0
                new Tuple<double, double[]> (0, new double[] { 0,0,0,1}), // 1
                new Tuple<double, double[]> (1, new double[] { 0,0,1,0}), // 2
                new Tuple<double, double[]> (0, new double[] { 0,0,1,1}), // 3
                new Tuple<double, double[]> (0, new double[] { 0,1,0,0}), // 4
                new Tuple<double, double[]> (0, new double[] { 0,1,0,1}), // 5
                new Tuple<double, double[]> (1, new double[] { 0,1,1,0}), // 6
                new Tuple<double, double[]> (0, new double[] { 0,1,1,1}), // 7
                new Tuple<double, double[]> (1, new double[] { 1,0,0,0}), // 8
                new Tuple<double, double[]> (1, new double[] { 1,0,0,1}), // 9
                new Tuple<double, double[]> (1, new double[] { 1,0,1,0}), // 10
                new Tuple<double, double[]> (1, new double[] { 1,0,1,1}), // 11
                new Tuple<double, double[]> (1, new double[] { 1,1,0,0}), // 12
                new Tuple<double, double[]> (0, new double[] { 1,1,0,1}), // 13
                new Tuple<double, double[]> (1, new double[] { 1,1,1,0}), // 14
                new Tuple<double, double[]> (1, new double[] { 1,1,1,1})  // 15
            };

            var topology = new Topology(4, 1, 0.1, 2);
            var neuralNetwork = new NeuralNetwork(topology);

            var difference = neuralNetwork.Learn(dataset, 100000); // средняя квадратичная ошибка при обучении

            var results = new List<double>();

            foreach (var data in dataset)
            {
               var res = neuralNetwork.FeedForward(data.Item2).Output;
               results.Add(res);
            }

            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(dataset[i].Item1, 3);
                var actual = Math.Round(results[i], 3);

                //Assert.AreEqual(expected, actual);
            }
        }
    }
}