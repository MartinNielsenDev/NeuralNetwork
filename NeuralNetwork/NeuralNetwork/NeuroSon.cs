using System;
using NeuralNetwork.Adaline;

namespace NeuralNetwork.Son
{
    public class SelfOrganizingNode : NeuroNode
    {
        public double LearningRate;
        public SelfOrganizingNode(double learningRate)
        {
            LearningRate = learningRate;
        }
        public override void Run()
        {
            double total = 0;
            foreach (var link in InLinks)
            {
                total += Math.Pow(link.InNode.Value - link.Weight, 2);
            }
            Value = Math.Sqrt(total);
        }
    }
    public class SelfOrganizingLink : AdalineLink
    {
    }

    public class SelfOrganizingNetwork : AdalineNetwork
    {
        protected int columsCount;
        protected long currentIteration;
        protected int currentNeighborhoodSize;
        protected double finalLearningRate;
        protected double initialLearningRate;
        protected int initialNeighborhoodSize;
        protected NeuroNode[,] kohonenLayer;
        protected int neighborhoodReduceInterval;
        protected int rowsCount;
        protected long trainingIterations;
        protected int winnigCol;
        protected int winnigRow;
        public SelfOrganizingNetwork(int aInputNodesCount, int aRowCount, int aColCount,
            double aInitialLearningRate, double aFinalLearningRate,
            int aInitialNeighborhoodSize, int aNeighborhoodReduceInterval,
            long aTrainingIterationsCount)
        {
            nodesCount = 0;
            linksCount = 0;
            initialLearningRate = aInitialLearningRate;
            finalLearningRate = aFinalLearningRate;
            learningRate = aInitialLearningRate;
            initialNeighborhoodSize = aInitialNeighborhoodSize;
            neighborhoodReduceInterval = aNeighborhoodReduceInterval;
            trainingIterations = aTrainingIterationsCount;
            currentIteration = 0;
            nodesCount = aInputNodesCount;
            currentNeighborhoodSize = initialNeighborhoodSize;
            rowsCount = aRowCount;
            columsCount = aColCount;
            CreateNetwork();
        }
        public SelfOrganizingNetwork()
        {
            nodesCount = 0;
            linksCount = 0;
        }
        public SelfOrganizingNetwork(double[] neuralNetworkData) : base(neuralNetworkData)
        {
        }
        public int KohonenRowsCount
        {
            get { return rowsCount; }
        }
        public int KohonenColumsCount
        {
            get { return columsCount; }
        }
        public int CurrentNeighborhoodSize
        {
            get { return currentNeighborhoodSize; }
        }
        public NeuroNode[,] KohonenNode
        {
            get { return kohonenLayer; }
        }
        public int WinnigRow
        {
            get { return winnigRow; }
        }
        public int WinnigCol
        {
            get { return winnigCol; }
        }
        protected override NeuralNetworkType GetNetworkType()
        {
            return NeuralNetworkType.nntSON;
        }
        protected override void CreateNetwork()
        {
            nodes = new NeuroNode[NodesCount];
            linksCount = NodesCount * rowsCount * columsCount;
            kohonenLayer = new NeuroNode[rowsCount, columsCount];
            links = new NeuroLink[LinksCount];
            for (var i = 0; i < NodesCount; i++)
                nodes[i] = new InputNode();
            var curr = 0;
            for (var row = 0; row < rowsCount; row++)
                for (var col = 0; col < columsCount; col++)
                {
                    kohonenLayer[row, col] = new SelfOrganizingNode(learningRate);
                    for (var i = 0; i < NodesCount; i++)
                    {
                        links[curr] = new SelfOrganizingLink();
                        nodes[i].LinkTo(kohonenLayer[row, col], links[curr]);
                        curr++;
                    }
                }
        }
        protected override int GetInputNodesCount()
        {
            return NodesCount;
        }
        protected override NeuroNode GetInputNode(int index)
        {
            if ((index >= InputNodesCount) || (index < 0))
                throw new ENeuroException("InputNode index out of bounds.");
            return nodes[index];
        }
        protected override NeuroNode GetOutputNode(int index)
        {
            return null;
        }
        protected override int GetOutPutNodesCount()
        {
            return 0;
        }
        public override void Epoch(int epoch)
        {
            currentIteration++;
            learningRate = initialLearningRate -
                           ((currentIteration / (double)trainingIterations) * (initialLearningRate - finalLearningRate));
            if (((((currentIteration + 1) % neighborhoodReduceInterval) == 0) && (currentNeighborhoodSize > 0)))
                currentNeighborhoodSize--;
        }
        protected override double GetNodeError()
        {
            return 0;
        }
        protected override void SetNodeError(double value)
        {
            //Cannot set the errors. Nothing is here....
        }
        public override void Load(double[] loadData)
        {
            initialLearningRate = ExtractDataFromArray(loadData);
            finalLearningRate = ExtractDataFromArray(loadData);
            initialNeighborhoodSize = Convert.ToInt32(ExtractDataFromArray(loadData));
            neighborhoodReduceInterval = Convert.ToInt32(ExtractDataFromArray(loadData));
            trainingIterations = Convert.ToInt64(ExtractDataFromArray(loadData));
            rowsCount = Convert.ToInt32(ExtractDataFromArray(loadData));
            columsCount = Convert.ToInt32(ExtractDataFromArray(loadData));
            base.Load(loadData);

            for (var r = 0; r < rowsCount; r++)
            {
                for (var c = 0; c < columsCount; c++)
                {
                    kohonenLayer[r, c].Load(loadData);
                }
            }
        }
        public override void Run()
        {
            var minValue = double.PositiveInfinity;
            LoadInputs();
            for (var row = 0; row < rowsCount; row++)
                for (var col = 0; col < columsCount; col++)
                {
                    kohonenLayer[row, col].Run();
                    var nodeValue = kohonenLayer[row, col].Value;
                    if (nodeValue < minValue)
                    {
                        minValue = nodeValue;
                        winnigRow = row;
                        winnigCol = col;
                    }
                }
        }
    }
}