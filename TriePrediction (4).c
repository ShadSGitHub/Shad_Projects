// Shady Saleh
// NID: sh430341

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "TriePrediction.h"

void stripPunctuators(char *str);
TrieNode *insert(TrieNode *root, char *str);
TrieNode *createTrieNode(void);
int prefixHelp(TrieNode *root, int counter);
void frequency(TrieNode *root, char *buffer, int k, int *max, char *holder);
void predict(TrieNode *root, char *str, int i);

// Helper function called by printTrie(). (Credit: Dr. S.)
void printTrieHelper(TrieNode *root, char *buffer, int k)
{
	int i;

	if (root == NULL)
		return;

	if (root->count > 0)
		printf("%s (%d)\n", buffer, root->count);

	buffer[k + 1] = '\0';

	for (i = 0; i < 26; i++)
	{
		buffer[k] = 'a' + i;

		printTrieHelper(root->children[i], buffer, k + 1);
	}

	buffer[k] = '\0';
}

// If printing a subtrie, the second parameter should be 1; otherwise, if
// printing the main trie, the second parameter should be 0. (Credit: Dr. S.)
void printTrie(TrieNode *root, int useSubtrieFormatting)
{
	char buffer[1026];

	if (useSubtrieFormatting)
	{
		strcpy(buffer, "- ");
		printTrieHelper(root, buffer, 2);
	}
	else
	{
		strcpy(buffer, "");
		printTrieHelper(root, buffer, 0);
	}
}

// (credit: Dr. S.)
// it has modification to take
// account for the subtrie
TrieNode *buildTrie(char *filename)
{
	TrieNode *root = NULL;
	char buffer[1024];
	char prev[1024] ;
	FILE *ifp;
	TrieNode *temp;
	int stop = 0;
	int flag = 0;
	int index = 0;

	strcpy(prev, "");
	ifp = fopen(filename, "r");
	if (ifp == NULL)
	{
		return NULL;
	}

	// the looping for the build while taking account the subtrie with two flag variables
	while (fscanf(ifp, "%s", buffer) != EOF)
	{
		index = strlen(buffer) - 1;
		if (buffer[index] == '?' || buffer[index] == '!' || buffer[index] == '.' )
		{
			stop = 1;
		}
		else
		{
			stop = 0;
		}

		stripPunctuators(buffer);
		root = insert(root, buffer);
		
		if( prev[0]!= '\0')
		{
			temp = getNode(root,prev);

			if(!flag)
			{
				temp->subtrie = insert(temp->subtrie,buffer);
				flag = 0;
			}
			else
			{
				flag = 0;
			}

			if(stop)
			{
				stop = 0;
				flag = 1;
			}
		}
	
		strcpy(prev,buffer);
	}

	fclose(ifp);

	return root;
}

int processInputFile(TrieNode *root, char *filename)
{
	char buffer[1024];
	int looper = 0;
	FILE *ifp;
	TrieNode *temp;
	TrieNode *tempNode = NULL;
	int x = 0 ;

	ifp = fopen(filename, "r");
	if (ifp == NULL)
	{
		return 1;
	}
	
	if(root == NULL)
	{
		return 1;
	}

	while (fscanf(ifp, "%s", buffer) != EOF)
	{
		if(buffer[0] == '@')
		{
			fscanf(ifp, "%s", buffer);
			fscanf(ifp, "%d", &looper);
			
			// the looping for text prediction.
			x = 0;
			while(x < (looper + 1))
			{
				
				if( (!containsWord(root, buffer) ) || (!getNode(root, buffer)->subtrie) || x == looper)
				{
					printf("%s", buffer);
				}
				else
				{
					printf("%s ", buffer);
				}
				
				if(!containsWord(root, buffer) || (!getNode(root, buffer)->subtrie) )
				{
					break;
				}

				getMostFrequentWord(getNode(root, buffer)->subtrie, buffer);

				x++;
			}

			printf("\n");
		}
		else if(buffer[0] == '!')
		{
			printTrie(root,0);
		}
		else if(containsWord(root, buffer))
		{
			printf("%s\n",buffer);
			temp = getNode(root,buffer);

			if(getNode(root,buffer)->subtrie)
			{
				printTrie(getNode(root,buffer)->subtrie,1);
			}
			else
			{
				printf("(EMPTY)\n");
			}
		}	
		else
		{
			printf("%s\n(INVALID STRING)\n", buffer);
		}
	}
	
	fclose(ifp);

	return 0;
}

// (Credit: Dr. S.)
// with modification
TrieNode *destroyTrie(TrieNode *root)
{
	int i;

    if(root == NULL) 
	{
		return NULL;
	}   
    
	if(root->subtrie != NULL)
	{
		destroyTrie(root->subtrie);
	}

    for (i = 0; i < 26; i++)
	{
       destroyTrie(root->children[i]);

	}

    free(root);

	return NULL;
}

TrieNode *getNode(TrieNode *root, char *str)
{
	int len = strlen(str);
	int i;
	TrieNode *temp;

	if(root == NULL)
	{
		return NULL;
	}

	if(str ==  NULL)
	{
		return NULL;
	}

	temp = root;

	// traversing the trie by each given letter then check if a word as been reached
	for(i = 0; i < len; i++)
	{
		if(temp->children[tolower(str[i]) - 'a'] != NULL)
		{
			temp = temp->children[tolower(str[i]) - 'a'];
		}
		else
		{
			return NULL;
		}
	}

	if(temp->count >= 1)
	{
		return temp;
	}

	return NULL;

}

// (credit: Dr. S.)
// based on print function
// with modification
void getMostFrequentWord(TrieNode *root, char *str)
{
	char buffer[1026];
	int max = 0;
	int x;

	if(str == NULL)
	{
		return;
	}

	memset(str,'\0',sizeof(str));

	if(root == NULL)
	{
		return;
	}
	
	strcpy(buffer, "");
	frequency(root, buffer, 0, &max ,str);
}

// (credit: Dr. S.)
// based on printhelper function
// with modification
void frequency(TrieNode *root, char *buffer, int k, int *max, char *holder)
{
	int i;
	int *temp;
	if (root == NULL)
		return;

	// instead of printing 
	// just get the new max and put the string in buffer in a temp variable
	if (root->count > 0)
	{
		if(root->count > *max)
		{
			*max = root->count;
			strcpy(holder,buffer);
		}
	}
	
	buffer[k + 1] = '\0';

	for (i = 0; i < 26; i++)
	{
		// adding two more arguments in the function call
		buffer[k] = 'a' + i;
		frequency(root->children[i], buffer, k + 1, max, holder);
	}

	buffer[k] = '\0';
}


int containsWord(TrieNode *root, char *str)
{
	if(root == NULL)
	{
		return 0;
	}

	if(str ==  NULL)
	{
		return 0;
	}

	// utilises already existing function 
	//because the return will always be either one or zero
	if(getNode(root,str))
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

int prefixCount(TrieNode *root, char *str)
{
	int i;
	int len = strlen(str);
	int sum = 0;
	TrieNode *temp = root;

	// defensive coding to avoid seg faults
    if(root == NULL) 
	{
		return 0;
	}   

	if(str == NULL)
	{
		return 0;
	}

    for (i = 0; i < len; i++)
	{
       if(temp == NULL)
		{
			return 0;
		}

		// defensive coding to avoid seg faults
		if(temp->children[tolower(str[i]) - 'a'] != NULL)
		{
			temp = temp->children[tolower(str[i]) - 'a'];
		}
		else
		{
			return 0;
		}
	}
	 
	// no need to set up variables just return the needed value right away
	return prefixHelp(temp,sum);
	
}

// recursion helper function
int prefixHelp(TrieNode *root, int counter)
{
	int i;

	if(root == NULL)
	{
		return counter;
	}
	
	for(i = 0; i < 26; i++)
	{
		counter =  prefixHelp(root->children[i],counter);
	}

	// increpemnt the counter after looping 
	counter = counter + root->count;

	return counter;
}

double difficultyRating(void)
{
	return 3.0;
}

double hoursSpent(void)
{
	return 11.0;
}

// (credit: Dr. S.)
// not modified 
TrieNode *createTrieNode(void)
{
	int i = 0;

	TrieNode *newNode = malloc(sizeof(TrieNode));
	if(newNode == NULL)
	{
		return NULL;
	}

	newNode->count = 0;

	for (i = 0; i < 26; i++)
	{
		newNode->children[i] = NULL;
	}

	newNode->subtrie = NULL;	
	return newNode;
}

// (credit: Dr. S.)
// not modified
TrieNode *insert(TrieNode *root, char *str)
{
	int i, index, len = strlen(str);
	TrieNode *wizard;

	if (root == NULL)
		root = createTrieNode();

	wizard = root;

	for (i = 0; i < len; i++)
	{
		index = tolower(str[i]) - 'a';

		if (wizard->children[index] == NULL)
			wizard->children[index] = createTrieNode();

		wizard = wizard->children[index];
	}

	wizard->count++;
	return root;
}


void stripPunctuators(char *str)
{
	int i;
	int k = 0;
	char suffer[1024]; 
	int len = strlen(str);

	// using a for loop to go through each character
	// if not a alphabetical character it automaticcaly loops to the next iteration
	for(i = 0; i < len; i++)
	{
		if(isalpha(str[i]))
		{
			suffer[k] = str[i];
			k++;
		}
	}

	suffer[k] = '\0';
 	strcpy(str,suffer);
}


int main(int argc, char **argv)
{
	TrieNode *root = buildTrie(argv[1]);
	
	processInputFile(root, argv[2]);
	
	root = destroyTrie(root);
	
	return 0;
}
