using System;
using NeuralNetwork;
using NeuralNetwork.Patterns;
using NeuralNetwork.Backprop;
using System.Drawing.Imaging;
using System.Drawing;

namespace BetterOverwatch
{
    public class BackPropNetwork : BackPropagationRPROPNetwork
    {
        public BackPropNetwork(int[] nodesInEachLayer) : base(nodesInEachLayer)
        {
        }
        private int OutputPatternIndex(Pattern pattern)
        {
            for (var i = 0; i < pattern.OutputsCount; i++)
                if (pattern.Output[i] == 1)
                    return i;
            return -1;
        }

        public int BestNodeIndex
        {
            get
            {
                int result = -1;
                double aMaxNodeValue = 0;
                double aMinError = double.PositiveInfinity;

                for (int i = 0; i < this.OutputNodesCount; i++)
                {
                    NeuroNode node = OutputNode(i);

                    if (node.Value > aMaxNodeValue || node.Value >= aMaxNodeValue && node.Error < aMinError)
                    {
                        aMaxNodeValue = node.Value;
                        aMinError = node.Error;
                        result = i;
                    }
                }

                return result;
            }
        }
        public static double[] CharToDoubleArray(Bitmap image)
        {
            BitmapData imageData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadWrite, image.PixelFormat);
            int imageFormatSize = Image.GetPixelFormatSize(imageData.PixelFormat);
            double[] result = new double[BetterOverwatchNetworks.matrixWidth * BetterOverwatchNetworks.matrixHeight];
            double xScale = (double)image.Width / (double)BetterOverwatchNetworks.matrixWidth;
            double yScale = (double)image.Height / (double)BetterOverwatchNetworks.matrixHeight;

            unsafe
            {
                byte* imageDataMemory = (byte*)imageData.Scan0;

                for (int x = 0; x < image.Width; ++x)
                {
                    for (int y = 0; y < image.Height; ++y)
                    {
                        int xResult = (int)(x / xScale);
                        int yResult = (int)(y / yScale);
                        int positionInMemory = (y * imageData.Stride) + (x * imageFormatSize / 8);

                        result[yResult * xResult + yResult] += Math.Sqrt(imageDataMemory[positionInMemory] * imageDataMemory[positionInMemory] + imageDataMemory[positionInMemory + 2] * imageDataMemory[positionInMemory + 2] + imageDataMemory[positionInMemory + 1] * imageDataMemory[positionInMemory + 1]);
                    }
                }
            }
            image.UnlockBits(imageData);

            return Scale(result);
        }
        private static double MaxOf(double[] src)
        {
            double res = double.NegativeInfinity;
            foreach (double d in src)
            {
                if (d > res) res = d;
            }
            return res;
        }
        private static double[] Scale(double[] src)
        {
            double max = MaxOf(src);
            if (max != 0)
            {
                for (int i = 0; i < src.Length; i++)
                {
                    src[i] = src[i] / max;
                }
            }
            return src;
        }
    }
    public class BetterOverwatchNetworks
    {
        public const int matrixWidth = 10;
        public const int matrixHeight = 10;
        public static BackPropNetwork mapsNN = new BackPropNetwork(new int[3] { matrixWidth * matrixHeight, (matrixWidth * matrixHeight + 26) / 2, 26 });
        public static BackPropNetwork teamSkillRatingNN = new BackPropNetwork(new int[3] { matrixWidth * matrixHeight, (matrixWidth * matrixHeight + 10) / 2, 10 });
        public static BackPropNetwork skillRatingNN = new BackPropNetwork(new int[3] { matrixWidth * matrixHeight, (matrixWidth * matrixHeight + 10) / 2, 10 });
        public static BackPropNetwork statsNN = new BackPropNetwork(new int[3] { matrixWidth * matrixHeight, (matrixWidth * matrixHeight + 50) / 2, 50 });
        public static BackPropNetwork heroNamesNN = new BackPropNetwork(new int[3] { matrixWidth * matrixHeight, (matrixWidth * matrixHeight + 26) / 2, 26 });

        public static void Load()
        {
            mapsNN.LoadFromArray(Data.mapsNNData);
            teamSkillRatingNN.LoadFromArray(Data.teamSkillRatingNNData);
            skillRatingNN.LoadFromArray(Data.skillRatingNNData);
            statsNN.LoadFromArray(Data.statsNNData);
            heroNamesNN.LoadFromArray(Data.heroNamesNNData);
        }
    }
}