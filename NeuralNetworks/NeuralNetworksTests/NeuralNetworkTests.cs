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
            var outputs = new double[] {0,0,1,0,0,0,1,0,1,1,1,1,1,0,1,1 };
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

            var topology = new Topology(4, 1, 0.1, 2);
            var neuralNetwork = new NeuralNetwork(topology);

            var difference = neuralNetwork.Learn(outputs, inputs, 100000); // средняя квадратичная ошибка при обучении

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

                //Assert.AreEqual(expected, actual);
            }
        }
    }
}