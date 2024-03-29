using System;
using NeuralNetwork.Adaline;

namespace NeuralNetwork.Backprop
{
    public class BackPropagationNode : FeedForwardNode
    {
        protected override double Transfer(double value)
        {
            return 1F / (1F + Math.Exp(-value)); //Sigmoid transfer 
        }
    }
    public class BackPropagationLink : NeuroLink
    {
        private double linkDelta;
        public BackPropagationLink()
        {
            DoAfterCreate();
        }
        public double LinkDelta
        {
            get { return GetLinkDelta(); }
            set { SetLinkDelta(value); }
        }
        protected virtual double GetLinkDelta()
        {
            return linkDelta;
        }
        protected virtual void DoAfterCreate()
        {
            Weight = Random(-1F, 1F);
            LinkDelta = 0;
        }

        protected virtual void SetLinkDelta(double delta)
        {
            linkDelta = delta;
        }
        public override void Load(double[] loadData)
        {
            base.Load(loadData);
            linkDelta = ExtractDataFromArray(loadData);
        }
        public override void UpdateWeight(double deltaWeight)
        {
            Weight = Weight + deltaWeight + ((BackPropagationOutputNode)OutNode).Momentum * LinkDelta;
            LinkDelta = deltaWeight;
        }
    }
    public class BackPropagationOutputNode : BackPropagationNode
    {
        private double learningRate;
        private double momentum;

        public BackPropagationOutputNode(double learningRate, double momentum)
        {
            LearningRate = learningRate;
            Momentum = momentum;
        }
        public double Momentum
        {
            get { return GetNodeMomentum(); }
            set { SetNodeMomentum(value); }
        }
        public double LearningRate
        {
            get { return GetNodeLearningRate(); }
            set { SetNodeLearningRate(value); }
        }
        protected virtual double ComputeError()
        {
            var nv = Value; //optimized for speed
            return nv * (1F - nv) * (Error - nv);
        }
        protected virtual double GetNodeMomentum()
        {
            return momentum;
        }
        protected virtual void SetNodeMomentum(double momentum)
        {
            this.momentum = momentum;
        }

        protected virtual double GetNodeLearningRate()
        {
            return learningRate;
        }
        protected virtual void SetNodeLearningRate(double learningRate)
        {
            this.learningRate = learningRate;
        }
        public override void Load(double[] loadData)
        {
            base.Load(loadData);

            momentum = ExtractDataFromArray(loadData);
            learningRate = ExtractDataFromArray(loadData);
        }
    }

    public class BackPropagationMiddleNode : BackPropagationOutputNode
    {
        public BackPropagationMiddleNode(double learningRate, double momentum) : base(learningRate, momentum)
        {
        }

        protected override double ComputeError()
        {
            double total = 0F;
            foreach (var link in OutLinks)
                total += link.WeightedOutError();
            var nv = Value;
            return nv * (1F - nv) * total;
        }
    }
    public class BackPropagationNetwork : AdalineNetwork
    {
        private readonly double momentum;
        protected int firstMiddleNode;
        protected int firstOutputNode;
        private int layersCount;
        private int[] nodesInLayer;

        public BackPropagationNetwork(double learningRate, double momentum, int[] nodesInEachLayer)
        {
            nodesCount = 0;
            linksCount = 0;
            layersCount = nodesInEachLayer.Length;
            nodesInLayer = new int[layersCount];
            for (var i = 0; i < layersCount; i++)
            {
                nodesInLayer[i] = nodesInEachLayer[i];
                nodesCount += nodesInLayer[i];
                if (i > 0)
                    linksCount += nodesInLayer[i - 1] * nodesInLayer[i];
            }
            this.learningRate = learningRate;
            this.momentum = momentum;
            CreateNetwork();
        }
        public BackPropagationNetwork()
        {
            nodesInLayer = null;
        }
        public BackPropagationNetwork(double[] neuralNetworkData) : base(neuralNetworkData)
        {
        }
        private int GetNodesInLayer(int index)
        {
            return nodesInLayer[index];
        }
        private NeuroNode GetMiddleNode(int index)
        {
            if ((index >= GetMiddleNodesCount()) || (index < 0))
                throw new ENeuroException("Middlenode index out of bounds.");
            //In case of Adaline an index always will be 0.
            return nodes[firstMiddleNode + index];
        }
        private int GetMiddleNodesCount()
        {
            return firstOutputNode - firstMiddleNode;
        }
        protected override NeuralNetworkType GetNetworkType()
        {
            return NeuralNetworkType.nntBackProp;
        }
        protected override int GetInputNodesCount()
        {
            return nodesInLayer[0];
        }
        protected override int GetOutPutNodesCount()
        {
            return nodesInLayer[layersCount - 1];
        }
        protected override NeuroNode GetOutputNode(int index)
        {
            if ((index >= OutputNodesCount) || (index < 0))
                throw new ENeuroException("OutputNode index out of bounds.");
            //In case of Adaline an index always will be 0.
            return nodes[firstOutputNode + index];
        }
        public virtual NeuroLink CreateLink()
        {
            return new BackPropagationLink();
        }
        protected override void CreateNetwork()
        {
            nodes = new NeuroNode[NodesCount];
            links = new NeuroLink[LinksCount];
            var cnt = 0;
            for (var i = 0; i < InputNodesCount; i++)
            {
                nodes[cnt] = new InputNode();
                cnt++;
            }

            firstMiddleNode = cnt;

            for (var i = 1; i < (layersCount - 1); i++)
                for (var j = 0; j < nodesInLayer[i]; j++)
                {
                    nodes[cnt] = new BackPropagationMiddleNode(LearningRate, momentum);
                    cnt++;
                }

            firstOutputNode = cnt;
            for (var i = 0; i < OutputNodesCount; i++)
            {
                nodes[cnt] = new BackPropagationOutputNode(LearningRate, momentum);
                cnt++;
            }

            for (var i = 0; i < LinksCount; i++)
                links[i] = CreateLink();
            cnt = 0;
            var l1 = 0;
            var l2 = firstMiddleNode;
            for (var i = 0; i < (layersCount - 1); i++)
            {
                for (var j = 0; j < nodesInLayer[i + 1]; j++)
                    for (var k = 0; k < nodesInLayer[i]; k++)
                    {
                        nodes[l1 + k].LinkTo(nodes[l2 + j], links[cnt]);
                        cnt++;
                    }
                l1 = l2;
                l2 += nodesInLayer[i + 1];
            }
        }
        public override void Load(double[] loadData)
        {
            layersCount = Convert.ToInt32(ExtractDataFromArray(loadData));
            nodesInLayer = new int[layersCount];

            for (var i = 0; i < layersCount; i++)
            {
                nodesInLayer[i] = Convert.ToInt32(ExtractDataFromArray(loadData));
            }
            base.Load(loadData);
        }
        public override void Run()
        {
            LoadInputs();
            for (var i = firstMiddleNode; i < NodesCount; i++)
                nodes[i].Run();
        }
        public int NodesInLayer(int index)
        {
            return GetNodesInLayer(index);
        }
    }
    public class EpochBackPropagationLink : BackPropagationLink
    {
        protected double linkEpoch;

        public double LinkEpoch
        {
            get { return GetLinkEpoch(); }
            set { SetLinkEpoch(value); }
        }
        protected virtual double GetLinkEpoch()
        {
            return linkEpoch;
        }
        protected void SetLinkEpoch(double epoch)
        {
            linkEpoch = epoch;
        }
        public override void Load(double[] loadData)
        {
            base.Load(loadData);
            linkEpoch = ExtractDataFromArray(loadData);
        }
        public override void UpdateWeight(double deltaWeight)
        {
            LinkEpoch = LinkEpoch + deltaWeight; //deltaWij  (2.17)
        }
        public override void Epoch(int epoch)
        {
            var momentum = ((BackPropagationOutputNode)OutNode).Momentum;
            var delta = LinkEpoch / epoch;
            Weight = Weight + delta + (momentum * LinkDelta); //Wji(n+1)=Wij(n)+deltaWij(n)+alpha*deltaWij(n-1);
            LinkDelta = delta;
            LinkEpoch = 0F;
        }
    }
    public class EpochBackPropagationNetwork : BackPropagationNetwork
    {
        public EpochBackPropagationNetwork(double learningRate, double momentum, int[] nodesInEachLayer)
            : base(learningRate, momentum, nodesInEachLayer)
        {
        }
        public EpochBackPropagationNetwork(double[] neuralNetworkData) : base(neuralNetworkData)
        {
        }
        public override NeuroLink CreateLink()
        {
            return new EpochBackPropagationLink();
        }
        protected override NeuralNetworkType GetNetworkType()
        {
            return NeuralNetworkType.nntEpochBackProp;
        }
        public override void Epoch(int epoch)
        {
            foreach (var link in links) link.Epoch(epoch);
        }
    }
    public class BackPropagationRPROPLink : EpochBackPropagationLink
    {
        private double decayTerm; //Decay
        private double eta;
        private double etaMinus; //learning decrease
        private double etaPlus; //learning increase
        private double initEta; //Initial Eta;
        private double maxDelta; //maximum learning
        private double minDelta; //minimum learning

        protected override void DoAfterCreate()
        {
            base.DoAfterCreate();
            etaPlus = 1.2F; //learning increase
            etaMinus = 0.5F; //learning decrease
            initEta = 0.05F; //Initial Eta;
            maxDelta = 50.0F; //maximum learning
            minDelta = 1.0E-6F; //minimum learning
            decayTerm = 0.0F; //Decay
            eta = initEta;
        }
        public override void Epoch(int epoch)
        {
            var delta = -LinkEpoch / epoch;
            var deltaOld = LinkDelta;
            var deltaProduct = deltaOld * delta;
            var deltaSign = Math.Sign(delta);
            delta = delta - decayTerm * Weight;
            if (deltaProduct >= 0)
            {
                if (deltaProduct > 0) eta = Math.Min(eta * etaPlus, maxDelta);
                Weight = Weight - deltaSign * eta;
                deltaOld = delta;
            }
            else if (deltaProduct < 0)
            {
                eta = Math.Max(eta * etaMinus, minDelta);
                deltaOld = 0.0;
            }
            LinkDelta = deltaOld;
            LinkEpoch = 0;
        }
    }
    public class BackPropagationRPROPNetwork : EpochBackPropagationNetwork
    {
        public BackPropagationRPROPNetwork(double[] neuralNetworkData) : base(neuralNetworkData)
        {
        }
        public BackPropagationRPROPNetwork(int[] nodesInEachLayer) : base(1, 0, nodesInEachLayer)
        {
        }
        public override NeuroLink CreateLink()
        {
            return new BackPropagationRPROPLink();
        }
    }
}