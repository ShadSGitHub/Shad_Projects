package core;
//@author Janet Baez
//University of Central Florida
//MasterMind2

import java.util.*;
import java.awt.Color;
import constants.Constants;

public class Codemaker implements ICodemaker
{
    //member variables     
    private Set<Color> secretCode;
    private ArrayList<Color> codemakerResponse;
    
    public Codemaker()
    {         
        secretCode = new HashSet();
        codemakerResponse = new ArrayList();
        
        generateSecretCode();
    }

    @Override
    public void generateSecreteCode() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

   /* @Override
    public void checkAttemptedCode(ArrayList<Color> attempt) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        
           
    }*/
    
    // Calling out method
    public void generateSecretCode()
    {
        Random random = new Random();
        
        while(getSecretCode().size()<Constants.MAX_PEGS)
        { 
           int index = random.nextInt(Constants.COLORS);
           Color selectedColor = Constants.codeColors.get(index);  
           getSecretCode().add(selectedColor);
        }
       
        System.out.println("\ngenerated the secret code!");
        for (Color color: getSecretCode())
        {
            System.out.println(color.toString());
        }
    }
  

    
    //Getter & setter @ secretCode   

    /**
     * @return the secretCode
     */
    public Set<Color> getSecretCode() {
        return secretCode;
    }

    /**
     * @param secretCode the secretCode to set
     */
    public void setSecretCode(Set<Color> secretCode) {
        this.secretCode = secretCode;
    }

    /**
     * @return the codemakerResponse
     */
    public ArrayList<Color> getCodemakerResponse() {
        return codemakerResponse;
    }

    /**
     * @param codemakerResponse the codemakerResponse to set
     */
    public void setCodemakerResponse(ArrayList<Color> codemakerResponse) {
        this.codemakerResponse = codemakerResponse;
    }
     
    
    
   public void checkAttemptedCode(ArrayList<Color> attempt)
   {
       codemakerResponse.clear();
        System.out.println("Codemaker is checking codebreaker attempt ");
        
        //Local variables
        ArrayList<Integer> x = new ArrayList<>();
        ArrayList<Integer> y = new ArrayList<>();
        ArrayList<Color> s = new ArrayList<>();

        int redPegs = 0;
        int whitePegs = 0;
        s.addAll(getSecretCode());
        
        //Checking if the codebreaker’s guess is 
        //exactly equal to the codemaker’s attempt 
        if(s.equals(attempt))
        {
            redPegs = 4;
            whitePegs = 0;
            
            for(int i = 0; i <redPegs; i++)
            {
                codemakerResponse.add(Color.RED);
            }
        }
        else
        {
            for(int i = 0; i < attempt.size();i++)
            {
                if(s.get(i).equals(attempt.get(i)))
                {
                    System.out.println("Found correct color in correct position");
                    redPegs++;
                    x.add(i);
                }
                else
                {
                    
                    if(s.contains(attempt.get(i)))
                    {
                        System.out.println("Found correct color in wrong position");
                        whitePegs++;
                        y.add(i);
                        System.out.println("Red  pegs: "+ redPegs+" white pegs "+whitePegs);
                    }
                }
                //System.out.println("Red  pegs: "+ redPegs+" white pegs "+whitePegs);
            }
            for(int i = 0; i < attempt.size(); i++)
            {
                if(x.contains(i))
                {
                    codemakerResponse.add(i,Color.RED);
                }
                else
                {
                    if(y.contains(i))
                    {
                        codemakerResponse.add(i,Color.WHITE);
                    }
                    else
                    {
                        codemakerResponse.add(i,null);
                    }
                }
                System.out.println("Codemaker's Response ");
                for(Color col : codemakerResponse)
                {
                   if(col != null)
                   { 
                        System.out.print(col+" \n");
                   }
                }
            }
        }
        
   }
}


