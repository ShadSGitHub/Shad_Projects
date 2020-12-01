// This is a basic template showing how to set up your Node and SkipList classes
// to be in the same file. For my test cases to work properly, you will need to
// implement your Node class here, in this file, outside of the SkipList class.
//
// Note: You will need to modify these class declarations to make your Node
// and SkipList containers generic and capable of holding any type of Comparable
// object.
//
// You should rename this file "SkipList.java" and remove this comment before
// submitting.
import java.io.*;
import java.util.*;

class Node<AnyType  extends Comparable<AnyType>>
{
	int height; 
	AnyType value;
	ArrayList<Node<AnyType>> next;

	// each node contructore that use height alos
	// initializes the node arraylist to nulls 
	Node(int height)
	{
		this.height = height;
		next =  new ArrayList<>();
		//next.add(null);
		for(int i = 0; i < height; i++)
		{
			next.add(null);
		}
	}

	// changed from AnyType to int
	Node(AnyType data, int height)
	{
		this.height = height;
		value = data;
		next = new ArrayList<>();
		//next.add(null);

		for(int i = 0; i < height; i++)
		{
			next.add(null);
		}
	}
	
	public AnyType value()
	{
		return value;
	}
	
	public int height()
	{
		return height;
	}

	public Node<AnyType> next(int level)
	{
		if(level < 0 || level > height -1)
		{
			return null;
		}	
		
		return next.get(level);
	}

	public void setNext(int level, Node<AnyType> node)
	{
		next.set(level, node);
	}
	
	

}

public class SkipList<AnyType extends Comparable<AnyType>>
{
	private Node<AnyType> head ;
	int nodeCounter = 0;
	
	// similar to the Node class the SkipList class
	// initializes with height variable;
	SkipList()
	{
		head = new Node<AnyType>(1);
	}
	
	SkipList(int height)
	{
		head = new Node<AnyType>(height);
		
	}
	
	public int size()
	{
		return nodeCounter;
	}
	
	public int height()
	{
		return head.height();
	}

	// public Node<AnyType> head()
	 public Node<AnyType> head()
	{
		return head;
	}	
	
	
	public void insert(AnyType data)
	{
		ArrayList<Node<AnyType>> trackNodes = new ArrayList<>();

		int headHeight = height();
		int maxH;
		int randH;
		Node<AnyType> lookAhead = head;
		Node<AnyType> temp;
		Node<AnyType> after;
		Node<AnyType> newNode;
		Node<AnyType> trak;
		int TrackSize;
		Node<AnyType> swap;
		int ward;
		int coin;
		
		// first Stage : locate the appropiate position for insertion
		// this stage also keeps track of nodes that require realignment of its nodes
		for(int i = headHeight - 1; i >= 0; i--)
		{
			temp = lookAhead.next.get(i);
			if(temp == null || (data.compareTo(temp.value) <= 0) )
			{
				trackNodes.add(lookAhead);
				
			}
			else if(data.compareTo(temp.value) > 0)
			{
				lookAhead = temp;
				i++;
			}	
		}

		
		maxH = getMaxHeight(nodeCounter);
		randH = generateRandomHeight(maxH);
		newNode = new Node<AnyType>(data,randH);
		
		Collections.reverse(trackNodes);
			
		// second stage: stiching the new node
		// using a swap like mechanism 
		for(int i = 0; i < randH; i++)
		{
			trak = trackNodes.get(i);
			swap = trak.next.get(i);
			trak.setNext(i,newNode);
			newNode.setNext(i,swap);
		}

		nodeCounter++;

		// final stage: checking for imbalance
		// increasing the height with chance
		if(headHeight < Math.log(nodeCounter)/Math.log(2))
		{
			ward = head.next.size() - 1;
			head.height++;
			head.next.add(null);
			temp = head.next.get(ward);


			while(temp!=null)
			{
				if(coinFlip())
				{
					temp.height++;
					temp.next.add(null);				
				}
				temp = temp.next.get(ward);
			}

			temp = head.next.get(ward);
			lookAhead = head;
			while(temp != null)
			{
				if(temp.height == head.height)
				{
					lookAhead.next.set(ward + 1,temp);
					lookAhead = temp;		
				}
				temp = temp.next.get(ward);	
			}
			
			
		}

	}
	
	private void print2()
	{
		Node<AnyType> temp = head;
		for(int i = head.height -1; i >= 0; i-- )
		{
			temp = head;
			System.out.println("level : "+ i);
			while(temp != null)
			{
				System.out.print(" "+ temp.value);
				temp = temp.next.get(i);
			}
			System.out.println();
		}
	}
	private void printSkip()
	{
		int headHeight = height();
		
		Node<AnyType> lookAhead = head;
		Node<AnyType> temp;
		
		for(int i = headHeight - 1; i >= 0; i--)
		{		
			System.out.println("level = " + i);
			temp = lookAhead;
			
			HashSet<Node<AnyType>> visited = new HashSet<>();
			while(temp != null)
			{
				if(visited.contains(temp)){
					System.out.print("loop detected (links to: "+temp.value+")");
					break;
				}
				
				System.out.print(temp.value + " --> ");
				visited.add(temp);
				temp = lookAhead.next.get(i);

			}
			
			System.out.println();
		}		
	}
	
	//literally the same except for
	// height is not random and is given instead
	public void insert(AnyType data, int height)	
	{
		ArrayList<Node<AnyType>> trackNodes = new ArrayList<>();

		int headHeight = height();
		int maxH;
		int randH;
		Node<AnyType> lookAhead = head;
		Node<AnyType> temp;
		Node<AnyType> after;
		Node<AnyType> newNode;
		Node<AnyType> trak;
		int TrackSize;
		Node<AnyType> swap;
		int ward;
		int coin;
		

		for(int i = headHeight - 1; i >= 0; i--)
		{
			temp = lookAhead.next.get(i);
			if(temp == null || (data.compareTo(temp.value) <= 0) )
			{
				trackNodes.add(lookAhead);
				
			}
			else if(data.compareTo(temp.value) > 0)
			{
				lookAhead = temp;
				i++;
			}	
		}

		
		// here is the difference
		// no random height
		randH = height;

		newNode = new Node<AnyType>(data,randH);
		Collections.reverse(trackNodes);
			
		for(int i = 0; i < randH; i++)
		{
			trak = trackNodes.get(i);
			swap = trak.next.get(i);
			trak.setNext(i,newNode);
			newNode.setNext(i,swap);
		}
		
		nodeCounter++;

		if(headHeight < Math.ceil(Math.log(nodeCounter)/Math.log(2)) )
		{
			ward = head.next.size() - 1;
			head.height++;
			head.next.add(null);
			temp = head.next.get(ward);


			while(temp!=null)
			{
				if(coinFlip())
				{
					temp.height++;
					temp.next.add(null);				
				}
				temp = temp.next.get(ward);
			}

			temp = head.next.get(ward);
			lookAhead = head;
			while(temp != null)
			{
				if(temp.height == head.height)
				{
					lookAhead.next.set(ward + 1,temp);
					lookAhead = temp;		
				}
				temp = temp.next.get(ward);	
			}
			
			
		}

	}
	
	// simple method to make sure a function exists	
	public boolean contains(AnyType data)
	{
		
		int headHeight = height();
		
		Node<AnyType> lookAhead = head;
		Node<AnyType> temp;
		
		// very similar to first stage of insert
		for(int i = headHeight - 1; i >= 0; i--)
		{
			temp = lookAhead.next.get(i);
			if(temp == null)
			{
				;
				
			}
			else if(data.compareTo(temp.value) < 0 )
			{
				;
				
			}
			else if(data.compareTo(temp.value) > 0)
			{
				lookAhead = temp;
				i++;
			}
			else if(data.compareTo(temp.value) == 0)
			{
				return true;
			
			}	
		}

	return false;

	}
	
	// similar to insert
	// but final stage is different
	public void delete(AnyType data)
	{
		ArrayList<Node<AnyType>> trackNodes = new ArrayList<>();

			int headHeight = height();
			int maxH;
			int randH;
			Node<AnyType> lookAhead = head;
			Node<AnyType> temp;
			Node<AnyType> after;
			Node<AnyType> delete = null;
			Node<AnyType> trak;
			int TrackSize;
			Node<AnyType> swap;
			int ward;
			int coin;
			
			// same as insert but using contains() mechinism
			for(int i = headHeight - 1; i >= 0; i--)
			{
				temp = lookAhead.next.get(i);
				if(temp == null || (data.compareTo(temp.value) <= 0) )
				{
					if(temp != null && (data.compareTo(temp.value) == 0))
					{
						delete = temp;
					}
					trackNodes.add(lookAhead);
					
				}
				else if(data.compareTo(temp.value) > 0)
				{
					lookAhead = temp;
					i++;
				}	
			}	
				
			// skips random number generator
			// just deletes
			if(delete != null)
			{
				int deleteHeight = delete.height;
				Collections.reverse(trackNodes);
					
				for(int i = 0; i < deleteHeight; i++)
				{	
					swap = delete.next.get(i);
					trak = trackNodes.get(i);
					trak.setNext(i,swap);
				}
				
				nodeCounter--;

					// making sure to trimm probabbly and to cut too much
					while(headHeight != getMaxHeight(nodeCounter) && nodeCounter > 1)
					{
						ward = head.next.size() - 1;
						head.height--;
						
						lookAhead = head.next.get(ward);
						temp = lookAhead;

						while(temp!=null)
						{
							temp.height--;
							lookAhead = temp.next.get(ward);
							temp.next.remove(ward);
							temp = lookAhead;
						}
						
						head.next.remove(ward);
						headHeight = height();;
					}

			}						

			
	}	

	public static double difficultyRating()
	{
		return 5.0;
	}

	public static double hoursSpent()
	{
		return 70.0;
	}
	
	private static int getMaxHeight(int n)
	{
		return (int)Math.ceil(   Math.log(n)/Math.log(2)    );
	}
	
	// "virtual coin flip"
	private static Boolean coinFlip()
	{
		int coin = (int)(Math.random() * 2);
		return (coin == 1);
	}
	
	// important method for generating random height
	private static int generateRandomHeight(int maxHeight)
	{
		int counter = 1;

		for (int i = 1; i < maxHeight; i++)
		{
			if (coinFlip())
			{
				counter++;
			}
			else
			{
				break;
			}
		}
		return counter;
	}

	// not a single test case tested for this method
	//very similar to contains method
	public Node<AnyType> get(AnyType data)
	{
		int headHeight = height();
		
		Node<AnyType> lookAhead = head;
		Node<AnyType> temp;
		
		// very similar to first stage of insert
		for(int i = headHeight - 1; i >= 0; i--)
		{
			temp = lookAhead.next.get(i);
			if(temp == null)
			{
				;
				
			}
			else if(data.compareTo(temp.value) < 0 )
			{
				;
				
			}
			else if(data.compareTo(temp.value) > 0)
			{
				lookAhead = temp;
				i++;
			}
			else if(data.compareTo(temp.value) == 0)
			{
				return temp;
			
			}	
		}
		return null;
	
	}
	
	
}









































