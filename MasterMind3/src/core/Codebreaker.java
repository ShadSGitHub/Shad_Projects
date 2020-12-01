package core;

//@author Janet Baez

import java.util.*;
import java.awt.*;
import constants.Constants;
import java.lang.reflect.Field;


public class Codebreaker implements ICodebreaker
{
    //member variables - Section 1
    private ArrayList<Color> codebreakerAttempt;
    
    // Section 3        
    public Codebreaker()
    {
        codebreakerAttempt = new ArrayList<>();
    }
    //Getter & setter for member variables - Section 2
    public ArrayList getCodebreakerAttempt()
    {
        consoleAttempt();
        return this.codebreakerAttempt;
    }
    public void setCodebreakerAttempt(ArrayList codebreakerAttempt)
    {
        this.codebreakerAttempt = codebreakerAttempt;
    }
    // Section 4
    public boolean checkCode(ArrayList<Color> attempt)
    {
        boolean z = false;
        for(Color color : attempt)
        {
            if(color != Color.RED)
            {
                z = true;
            }
        }
        if(z)
            {
                return false;
            }
        else
        {
            return true;
        }
    }
    
    
    private void consoleAttempt()
    {
        //Prompting each attempt
        int numberOfTimes = Game.getAttempt() + 1;
        System.out.println("\n*** Attempt "+numberOfTimes +"***");

        codebreakerAttempt.clear();
        Scanner scan =  new Scanner(System.in);

        System.out.print("\nEnter your colors in left to right order\n" +
        "Use BLUE, BLACK, ORANGE, WHITE, YELLOW, RED, GREEN, PINK: \n");
        
        //Looping until the user enters four valid colors with no duplicates
        int valid = 4;
        while(valid != 0)
        {
            String colorString = scan.next();
            Color color = stringToColor(colorString);
            
            if(!codebreakerAttempt.contains(color) && (Constants.codeColors.contains(color)))
            {
                System.out.println("You entered "+colorString);
                codebreakerAttempt.add(color);
                valid--;
                if(valid != 0)
                {
                    System.out.println("Next color");
                }
            } 
            else
            {
                System.out.println("Invalid color choice, try again");
                System.out.println("Next color");
            }
        }

        System.out.println("Codebreaker's attempt ");
        for(Color c :codebreakerAttempt)
        {
            System.out.print(c+"\n");
        }
    }
  
  private Color stringToColor(String colorString)
  {
      Color color;
      try
      {
          Field field = Class.forName("java.awt.Color").getField(colorString);
          color = (Color)field.get(null);
      }
      catch (Exception e)
      {
          color = null;
      }
      return color;
  }
  
  
}
