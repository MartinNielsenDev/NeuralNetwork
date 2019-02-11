using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using NeuralNetwork.Patterns;

namespace NeuralNetwork
{
    public enum NeuralNetworkType
    {
        nnAdaline,
        nntBackProp,
        nntSON,
        nntBAM,
        nntBAMSystem,
        nntEpochBackProp
    };

    public abstract class NeuroObject
    {
        private static readonly Random random = new Random();
        public static int loadDataIndex = 0;
        internal NeuroObject()
        {
        }
        public virtual void Epoch(int epoch)
        {
        }
        public virtual void Load(double[] loadData)
        {
        }
        public virtual void LoadFromArray(double[] neuralNetworkData)
        {
            loadDataIndex = 0;
            Load(neuralNetworkData);
        }
        public static double ExtractDataFromArray(double[] loadData)
        {
            double result = loadData[loadDataIndex];
            loadDataIndex++;

            return result;
        }
        public static int RoundToNextInt(double value)
        {
            var result = (int) Math.Round(value);
            if (value > 0)
            {
                if (value > result) result++;
            }
            else
            {
                if (value < result) result--;
            }
            return result;
        }
        public static double Random(double min, double max)
        {
            double result, aRange;
            if (min > max)
            {
                result = max;
                max = min;
                min = result;
            }
            if (min == max) return max;
            aRange = max - min;
            return random.NextDouble()*aRange + min;
        }
    }
    public class ENeuroException : ApplicationException
    {
        public ENeuroException(string message) : base(message)
        {
        }
    }
    public class NeuroLink : NeuroObject
    {
        protected NeuroNode inNode, outNode;
        protected double linkWeight;

        public NeuroLink()
        {
            inNode = null;
            outNode = null;
        }
        public double Weight
        {
            get { return GetLinkWeight(); }
            set { SetLinkWeight(value); }
        }
        public NeuroNode InNode
        {
            get { return inNode; }
        }
        public NeuroNode OutNode
        {
            get { return outNode; }
        }
        protected virtual double GetLinkWeight()
        {
            return linkWeight;
        }
        protected virtual void SetLinkWeight(double value)
        {
            linkWeight = value;
        }
        public override void Load(double[] loadData)
        {
            base.Load(loadData);
            linkWeight = ExtractDataFromArray(loadData);
        }
        public void SetInNode(NeuroNode node)
        {
            inNode = node;
        }
        public void SetOutNode(NeuroNode node)
        {
            outNode = node;
        }
        public virtual void UpdateWeight(double deltaWeight)
        {
            Weight += deltaWeight;
        }
        public virtual double WeightedInValue()
        {
            return InNode.Value*Weight;
        }
        public virtual double WeightedOutValue()
        {
            return OutNode.Value*Weight;
        }
        public virtual double WeightedInError()
        {
            return InNode.Error*Weight;
        }
        public virtual double WeightedOutError()
        {
            return OutNode.Error*Weight;
        }
    }
    [Serializable]
    public class NeuroLinkCollection : NeuroObjectCollection
    {
        public NeuroLinkCollection()
        {
        }
        public NeuroLinkCollection(NeuroLinkCollection value)
        {
            AddRange(value);
        }
        public NeuroLinkCollection(NeuroLink[] value)
        {
            AddRange(value);
        }
        public NeuroLink this[int index]
        {
            get { return ((NeuroLink) (List[index])); }
            set { List[index] = value; }
        }
        protected override NeuroObject CreateContainigObject()
        {
            return new NeuroLink();
        }
        public int Add(NeuroLink value)
        {
            return List.Add(value);
        }
        public void AddRange(NeuroLink[] value)
        {
            for (var i = 0; (i < value.Length); i = (i + 1))
            {
                Add(value[i]);
            }
        }
        public void AddRange(NeuroLinkCollection value)
        {
            for (var i = 0; (i < value.Count); i = (i + 1))
            {
                Add(value[i]);
            }
        }
        public bool Contains(NeuroLink value)
        {
            return List.Contains(value);
        }
        public void CopyTo(NeuroLink[] array, int index)
        {
            List.CopyTo(array, index);
        }
        public int IndexOf(NeuroLink value)
        {
            return List.IndexOf(value);
        }
        public void Insert(int index, NeuroLink value)
        {
            List.Insert(index, value);
        }
        public new CustomNeuroLinkEnumerator GetEnumerator()
        {
            return new CustomNeuroLinkEnumerator(this);
        }
        public void Remove(NeuroLink value)
        {
            List.Remove(value);
        }
        public class CustomNeuroLinkEnumerator : object, IEnumerator
        {
            private readonly IEnumerator baseEnumerator;
            private readonly IEnumerable temp;

            public CustomNeuroLinkEnumerator(NeuroLinkCollection mappings)
            {
                temp = mappings;
                baseEnumerator = temp.GetEnumerator();
            }
            public NeuroLink Current
            {
                get { return ((NeuroLink) (baseEnumerator.Current)); }
            }
            object IEnumerator.Current
            {
                get { return baseEnumerator.Current; }
            }
            bool IEnumerator.MoveNext()
            {
                return baseEnumerator.MoveNext();
            }
            void IEnumerator.Reset()
            {
                baseEnumerator.Reset();
            }
            public bool MoveNext()
            {
                return baseEnumerator.MoveNext();
            }
            public void Reset()
            {
                baseEnumerator.Reset();
            }
        }
    }
    public class NeuroNode : NeuroObject
    {
        private readonly NeuroLinkCollection outLinks;
        protected double nodeValue, nodeError;

        public NeuroNode()
        {
            InLinks = new NeuroLinkCollection();
            outLinks = new NeuroLinkCollection();
        }
        public NeuroLinkCollection InLinks { get; private set; }
        public NeuroLinkCollection OutLinks
        {
            get { return outLinks; }
        }
        public double Value
        {
            get { return GetNodeValue(); }
            set { SetNodeValue(value); }
        }
        public double Error
        {
            get { return GetNodeError(); }
            set { SetNodeError(value); }
        }
        protected virtual double GetNodeValue()
        {
            return nodeValue;
        }
        protected virtual void SetNodeValue(double value)
        {
            nodeValue = value;
        }
        protected virtual double GetNodeError()
        {
            return nodeError;
        }
        protected virtual void SetNodeError(double error)
        {
            nodeError = error;
        }
        public virtual void Run()
        {
        }
        public void LinkTo(NeuroNode toNode, NeuroLink link)
        {
            OutLinks.Add(link);
            toNode.InLinks.Add(link);
            link.SetInNode(this);
            link.SetOutNode(toNode);
        }
        public override void Load(double[] loadData)
        {
            base.Load(loadData);
            nodeValue = ExtractDataFromArray(loadData);
            nodeError = ExtractDataFromArray(loadData);
        }
    }
    public abstract class NeuralNetwork : NeuroNode
    {
        protected NeuroLink[] links;
        protected int linksCount;
        protected NeuroNode[] nodes;
        protected int nodesCount;

        public NeuralNetwork()
        {
            nodesCount = 0;
            linksCount = 0;
            nodes = null;
            links = null;
        }
        public NeuralNetwork(double[] neuralNetworkData)
        {
            nodesCount = 0;
            linksCount = 0;
            nodes = null;
            links = null;
            LoadFromArray(neuralNetworkData);
        }
        public NeuralNetworkType NetworkType
        {
            get { return GetNetworkType(); }
        }
        public int NodesCount
        {
            get { return nodesCount; }
        }
        public int LinksCount
        {
            get { return linksCount; }
        }
        public int InputNodesCount
        {
            get { return GetInputNodesCount(); }
        }
        public int OutputNodesCount
        {
            get { return GetOutPutNodesCount(); }
        }

        private void CheckNetworkType(double[] loadData)
        {
            NeuralNetworkType nt = 0;
            nt = (NeuralNetworkType)Convert.ToInt32(ExtractDataFromArray(loadData));

            if (NetworkType != nt)
                throw new ENeuroException("Cannot load data. Invalid format.");
        }
        private void SaveNetworkType(BinaryWriter binaryWriter, List<double> saveData)
        {
            saveData.Add((int)NetworkType);
            binaryWriter.Write((int)NetworkType);
        }
        protected virtual void CreateNetwork()
        {
        }
        protected virtual void LoadInputs()
        {
        }
        protected abstract NeuralNetworkType GetNetworkType();
        protected abstract int GetInputNodesCount();
        protected abstract int GetOutPutNodesCount();
        protected abstract NeuroNode GetInputNode(int index);
        protected abstract NeuroNode GetOutputNode(int index);

        public override void Epoch(int epoch)
        {
            foreach (var node in nodes) node.Epoch(epoch);
            foreach (var link in links) link.Epoch(epoch);
            base.Epoch(epoch);
        }
        public override void Load(double[] loadData)
        {
            CheckNetworkType(loadData);
            nodesCount = Convert.ToInt32(ExtractDataFromArray(loadData));
            linksCount = Convert.ToInt32(ExtractDataFromArray(loadData));
            CreateNetwork();

            foreach (var node in nodes)
            {
                node.Load(loadData);
            }
            foreach (var link in links)
            {
                link.Load(loadData);
            }
        }
        public NeuroNode InputNode(int index)
        {
            return GetInputNode(index);
        }
        public NeuroNode OutputNode(int index)
        {
            return GetOutputNode(index);
        }
    }
    public class FeedForwardNode : NeuroNode
    {
        protected virtual double Transfer(double value)
        {
            return value;
        }
        public override void Run()
        {
            double total = 0;
            foreach (var link in InLinks) total += link.WeightedInValue();
            Value = Transfer(total);
        }
    }
    public class InputNode : NeuroNode
    {
    }
    public class BiasNode : InputNode
    {
        public BiasNode(double biasValue)
        {
            nodeValue = biasValue;
        }
        protected override void SetNodeValue(double value)
        {
            //Cannot change value of BiasNode
        }
    }
}