package core;
//@author Janet Baez

import java.util.*;
import java.awt.Color;
import constants.Constants;
import java.lang.System;

public class Game implements IGame
{
    //Section 1
    private static int attempt;
    private Codebreaker codebreaker;
    private Codemaker codemaker;
    
    //Section 3
    public Game() 
    {
        attempt = 0;
        codebreaker = new Codebreaker();
        codemaker = new Codemaker();  
        //play();
    }
    //Section 4
    public void play()
    {
        while(true)
        {
            if(attempt < Constants.MAX_ATTEMPTS)
            {
                ArrayList<Color> codeBreakerAttempt = codebreaker.getCodebreakerAttempt();
                codemaker.checkAttemptedCode(codeBreakerAttempt);
                boolean success = codebreaker.checkCode(codemaker.getCodemakerResponse());
                
                if(success)
                {
                    System.out.println("You guessed it!");
                    break;
                }
            }
            else
            {
                System.out.println("Codemaker wins");
                break;
            }
            attempt++;
        }
    }
    //Section 2

    /**
     * @return the attempt
     */
    public static int getAttempt() {
        return attempt;
    }

    /**
     * @param attempt the attempt to set
     */
    public void setAttempt(int attempt) {
        this.attempt = attempt;
    }

    /**
     * @return the codebreaker
     */
    public Codebreaker getCodebreaker() {
        return codebreaker;
    }

    /**
     * @param codebreaker the codebreaker to set
     */
    public void setCodebreaker(Codebreaker codebreaker) {
        this.codebreaker = codebreaker;
    }

    /**
     * @return the codemaker
     */
    public Codemaker getCodemaker() {
        return codemaker;
    }

    /**
     * @param codemaker the codemaker to set
     */
    public void setCodemaker(Codemaker codemaker) {
        this.codemaker = codemaker;
    }
   
}